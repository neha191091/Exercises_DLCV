import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torchvision.models as models
import math

def get_output_size_Conv2d(input_size, kernel_size, stride=1, padding = 0, dilation=1):
    #`H_{out} = floor((H_{in}  + 2 * padding[0] - dilation[0] * (kernel\_size[0] - 1) - 1) / stride[0] + 1)
    output_size = math.floor((input_size + 2*padding - dilation*(kernel_size - 1))/stride + 1)
    return output_size

def get_output_size_maxpool2d(input_size, kernel_size, stride=1, padding=0 , dilation=1):
    #`H_{out} = floor((H_{in}  + 2 * padding[0] - dilation[0] * (kernel\_size[0] - 1) - 1) / stride[0] + 1)`
    output_size = math.floor((input_size + 2*padding - dilation*(kernel_size - 1))/stride + 1)
    return output_size


class SegmentationNetwork(nn.Module):

    def __init__(self, pretr_net = 'alexnet', gpu_device = 0, netstride = 32):
        super(SegmentationNetwork, self).__init__()

        ############################################################################
        #                             YOUR CODE                                    #
        ############################################################################
        #pretr_net = 'alexnet'
        #print ('gpu dev for params', gpu_device)
        self.netstride = 32
        if(pretr_net == 'alexnet'):
            alexnet = models.alexnet(pretrained=True).cuda(gpu_device)
            self.features = alexnet.features

            # classifier
            alexnet_classifier_children = alexnet.classifier.named_children()
            classifier = nn.Sequential()

            count = 0
            for name, module in alexnet_classifier_children:
                if (isinstance(module, nn.Linear)):
                    count += 1
                    #print ('Gotta Linear')
                    if (count == 1):
                        first_conv_layer_wts = module.state_dict()["weight"].view(4096, 256, 6, 6)
                        first_conv_layer_biases = module.state_dict()["bias"]
                        first_conv_layer = nn.Conv2d(256, 4096, 6, 6)
                        first_conv_layer.load_state_dict({"weight": first_conv_layer_wts, "bias": first_conv_layer_biases})
                        classifier.add_module(name, first_conv_layer)
                    elif (count == 2):
                        second_conv_layer_wts = module.state_dict()["weight"].view(4096, 4096, 1, 1)
                        second_conv_layer_biases = module.state_dict()["bias"]
                        second_conv_layer = nn.Conv2d(4096, 4096, 1, 1)
                        second_conv_layer.load_state_dict({"weight": second_conv_layer_wts, "bias": second_conv_layer_biases})
                        classifier.add_module(name, second_conv_layer)
                else:
                    classifier.add_module(name=name, module=module)
            #classifier.add_module('last_layer', nn.Conv2d(4096, 23, 1, 1))
            self.classifier = classifier.cuda(gpu_device)
            classification_layer =  nn.Conv2d(4096, 23, 1, 1)

            # 16Stride


            # Pooling - Conv Layer 5
            #:math:`H_{out} = floor((H_{in}  + 2 * padding[0] - dilation[0] * (kernel\_size[0] - 1) - 1) / stride[0] + 1)`

            # Restructure the feature net to have two parts - self.features_1 - outputs conv 4 , self.feature_2 - outputs  the rest
            features_1 = nn.Sequential()
            features_2 = nn.Sequential()
            for name, module in self.features.named_children():
                if (isinstance(module, nn.Conv2d)):
                    count += 1
                    #print ('Gotta Linear')
                if (count <= 4):
                    features_1.add_module(name, module)
                else:
                    features_2.add_module(name, module)
            self.features_1 = features_1.cuda(gpu_device)
            self.features_2 = features_2.cuda(gpu_device)

            #A 1x1 conv net for after conv_4
            self.after_conv_4 = nn.Sequential(nn.Dropout(),nn.Conv2d(256,23,1,1)).cuda(gpu_device)


            #Alexnet features
            #nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            #nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=3, stride=2),
            #nn.Conv2d(64, 192, kernel_size=5, padding=2),
            #nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=3, stride=2),
            #nn.Conv2d(192, 384, kernel_size=3, padding=1),
            #nn.ReLU(inplace=True),
            #nn.Conv2d(384, 256, kernel_size=3, padding=1),
            #nn.ReLU(inplace=True),
            #nn.Conv2d(256, 256, kernel_size=3, padding=1),
            #nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=3, stride=2),

            input_size = 240

            lout_conv_1 = get_output_size_Conv2d(input_size,kernel_size=11, stride=4, padding=2)
            lout_maxpool_1 = get_output_size_maxpool2d(lout_conv_1, kernel_size=3, stride=2)
            lout_conv_2 = get_output_size_Conv2d(lout_maxpool_1,kernel_size=5, padding=2)
            lout_maxpool_2 = get_output_size_maxpool2d(lout_conv_2, kernel_size=3, stride=2)
            lout_conv_3 = get_output_size_Conv2d(lout_maxpool_2,  kernel_size=3, padding=1)
            lout_conv_4 = get_output_size_Conv2d(lout_conv_3, kernel_size=3, padding=1)

            # Deconv_1
            Lout = lout_conv_4
            Lin = 1
            stride = netstride
            kernel_size = Lout - (Lin - 1) * stride
            print(kernel_size, Lout, Lin, stride)
            kernel_size = int(kernel_size)
            self.deconvolution_16_1 = nn.Sequential(
                nn.ConvTranspose2d(23, 23, kernel_size, stride=netstride),
                nn.Dropout()).cuda(gpu_device)



            # Deconv_2
            Lout = 240
            Lin = lout_conv_4
            stride = netstride
            kernel_size = Lout - (Lin - 1) * stride
            print(kernel_size, Lout, Lin, stride)
            kernel_size = int(kernel_size)
            self.deconvolution_16_2 = nn.ConvTranspose2d(23, 23, kernel_size, stride=netstride).cuda(gpu_device)

        elif(pretr_net == 'vgg16'):
            vgg16 = models.vgg16(pretrained=True).cuda(gpu_device)
            self.features = vgg16.features

            #classifier
            vgg_classifier_children = vgg16.classifier.named_children()
            classifier = nn.Sequential()

            count = 0
            for name, module in vgg_classifier_children:
                if(isinstance(module,nn.Linear)):
                    count += 1
                    if(count == 1):
                        first_conv_layer_wts = module.state_dict()["weight"].view(4096,512,7,7)
                        first_conv_layer_biases = module.state_dict()["bias"]
                        first_conv_layer = nn.Conv2d(512,4096,7,7).load_state_dict({"weight": first_conv_layer_wts, "bias": first_conv_layer_biases})
                        classifier.add_module(name,first_conv_layer)
                    elif(count == 2):
                        second_conv_layer_wts = module.state_dict()["weight"].view(4096, 4096, 1, 1)
                        second_conv_layer_biases = module.state_dict()["bias"]
                        second_conv_layer = nn.Conv2d(4096, 4096, 1, 1).load_state_dict(
                            {"weight": second_conv_layer_wts, "bias": second_conv_layer_biases})
                        classifier.add_module(name, second_conv_layer)
                else:
                    classifier.add_module(name=name,module=module)
            #classifier.add_module('last_layer', nn.Conv2d(4096,23,1,1))
            self.classifier = classifier.cuda(gpu_device)
            classification_layer = nn.Conv2d(4096, 23, 1, 1)
        else:
            print('Initialization failed, supply the correct parameter {''alexnet'', ''vgg16''}')
            return
        self.classification_layer = classification_layer.cuda(gpu_device)

        #32Stride
        Lout = 240
        Lin = 1
        stride = netstride
        kernel_size = Lout - (Lin - 1) * stride
        kernel_size = int(kernel_size)
        self.deconvolution = nn.ConvTranspose2d(23, 23, kernel_size, stride=netstride).cuda(gpu_device)

        self.pretr_net = pretr_net


    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        ############################################################################
        #                             YOUR CODE                                    #
        ############################################################################
        #out_conv_encoder = self.conv_encoder(x)
        #out_conv_encoder = out_conv_encoder.view(out_conv_encoder.size(0), -1)
        #out_fc_encoder = self.fc_encoder(out_conv_encoder)
        #out = self.final_fc(out_fc_encoder)
        if(self.pretr_net == 'alexnet' and self.netstride == 8):
            features_1_out = self.features_1(x)
            features_2_out = self.features_2(features_1_out)
            after_conv_4_out = self.after_conv_4(features_1_out)
            classifier_out = self.classifier(features_2_out)
            deconv_1_out = self.deconvolution_16_1(classifier_out)
            deconv_2_out = self.deconvolution_16_2(after_conv_4_out + deconv_1_out)
            return deconv_2_out
        else:
            x = self.features(x)
            x = self.classifier(x)
            x = self.classification_layer(x)
            x = self.deconvolution(x)
            return x

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print 'Saving model... %s' % path
        torch.save(self, path)
