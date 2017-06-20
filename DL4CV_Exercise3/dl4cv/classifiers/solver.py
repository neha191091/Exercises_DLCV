from random import shuffle
import numpy as np

import torch
from torch.autograd import Variable


class Solver(object):
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self, optim=torch.optim.Adam, optim_args={},
                 loss_func=torch.nn.CrossEntropyLoss):
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func()

        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

    def train(self, model, train_loader, val_loader, num_epochs=10, log_nth=0, gpu_device=0):
        """
        Train a given model with the provided data.

        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        - num_epochs: total number of training epochs
        - log_nth: log training accuracy and loss every nth iteration
        """
        optim = self.optim(model.parameters(), **self.optim_args)
        self._reset_histories()
        iter_per_epoch = len(train_loader)

        print 'START TRAIN.'
        ############################################################################
        # TODO:                                                                    #
        # Write your own personal training method for our solver. In Each epoch    #
        # iter_per_epoch shuffled training batches are processed. The loss for     #
        # each batch is stored in self.train_loss_history. Every log_nth iteration #
        # the loss is logged. After one epoch the training accuracy of the last    #
        # mini batch is logged and stored in self.train_acc_history.               #
        # We validate at the end of each epoch, log the result and store the       #
        # accuracy of the entire validation set in self.val_acc_history.           #
        #
        # Your logging should like something like:                                 #
        #   ...                                                                    #
        #   [Iteration 700/4800] TRAIN loss: 1.452                                 #
        #   [Iteration 800/4800] TRAIN loss: 1.409                                 #
        #   [Iteration 900/4800] TRAIN loss: 1.374                                 #
        #   [Epoch 1/5] TRAIN acc/loss: 0.560/1.374                                #
        #   [Epoch 1/5] VAL   acc/loss: 0.539/1.310                                #
        #   ...                                                                    #
        ############################################################################
        # Loss and Optimizer
        for epoch in range(num_epochs):
            total_predictions = 0
            correct_predictions = 0
            for itr, (x, y) in enumerate(train_loader):
                #inputs = Variable(x)
                #labels = Variable(y)

                cuda_batch = x.cuda(gpu_device)
                cuda_targets = y.cuda(gpu_device)
                inputs = Variable(cuda_batch, requires_grad=False)
                labels = Variable(cuda_targets, requires_grad=False)

                optim.zero_grad()
                pred_labels = model.forward(inputs)
                loss = self.loss_func(pred_labels, labels)
                self.train_loss_history.append(loss.data[0])
                total_predictions += labels.size(0)

                y_pred = np.argmax(pred_labels.data.cpu().numpy(), axis=1)
                # print(y_pred)
                y_actual = labels.data.cpu().numpy()
                # print (y_actual)
                pred_acc = np.sum(np.array(y_pred) == np.array(y_actual))

                #_, predicted = torch.max(pred_labels.data, 1)
                #bool_arr = (predicted.cpu() == labels)
                correct_predictions += pred_acc
                loss.backward()
                optim.step()
                if (itr + 1) % log_nth == 0:
                    print('[Iteration %d/%d] TRAIN loss: %.4f' % (itr + 1, iter_per_epoch, loss.data[0]))

            train_acc = float(correct_predictions) / total_predictions
            self.train_acc_history.append(train_acc)
            print('[Epoch %d/%d] TRAIN acc/loss: %.4f/%.4f' % (epoch + 1, num_epochs, train_acc, loss.data[0]))

            total_predictions = 0
            correct_predictions = 0
            for itr, (x, y) in enumerate(val_loader):
                #inputs = Variable(x)
                #labels = Variable(y)

                cuda_batch = x.cuda(gpu_device)
                cuda_targets = y.cuda(gpu_device)
                inputs = Variable(cuda_batch, requires_grad=False)
                labels = Variable(cuda_targets, requires_grad=False)

                pred_labels = model.forward(inputs)
                loss = self.loss_func(pred_labels, labels)
                total_predictions += labels.size(0)
                #_, predicted = torch.max(pred_labels.data, 1)
                # correct_predictions += (predicted == labels).sum()

                y_pred = np.argmax(pred_labels.data.cpu().numpy(), axis=1)
                # print(y_pred)
                y_actual = labels.data.cpu().numpy()
                # print (y_actual)
                pred_acc = np.sum(np.array(y_pred) == np.array(y_actual))


                correct_predictions += pred_acc
            val_acc = float(correct_predictions) / total_predictions
            self.val_acc_history.append(val_acc)
            print('[Epoch %d/%d] VAL acc/loss: %.4f/%.4f'
                  % (epoch + 1, num_epochs, val_acc, loss.data[0]))

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        print 'FINISH.'
    def train_segmentation(self, model, train_loader, val_loader, num_epochs=10, log_nth=0, gpu_device=0):
        """
        Train a given model with the provided data.

        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        - num_epochs: total number of training epochs
        - log_nth: log training accuracy and loss every nth iteration
        """
        optim = self.optim(model.parameters(), **self.optim_args)
        self._reset_histories()
        iter_per_epoch = len(train_loader)

        print 'START TRAIN.'
        ############################################################################
        # TODO:                                                                    #
        # Write your own personal training method for our solver. In Each epoch    #
        # iter_per_epoch shuffled training batches are processed. The loss for     #
        # each batch is stored in self.train_loss_history. Every log_nth iteration #
        # the loss is logged. After one epoch the training accuracy of the last    #
        # mini batch is logged and stored in self.train_acc_history.               #
        # We validate at the end of each epoch, log the result and store the       #
        # accuracy of the entire validation set in self.val_acc_history.           #
        #
        # Your logging should like something like:                                 #
        #   ...                                                                    #
        #   [Iteration 700/4800] TRAIN loss: 1.452                                 #
        #   [Iteration 800/4800] TRAIN loss: 1.409                                 #
        #   [Iteration 900/4800] TRAIN loss: 1.374                                 #
        #   [Epoch 1/5] TRAIN acc/loss: 0.560/1.374                                #
        #   [Epoch 1/5] VAL   acc/loss: 0.539/1.310                                #
        #   ...                                                                    #
        ############################################################################
        # Loss and Optimizer
        for epoch in range(num_epochs):
            total_predictions = 0
            correct_predictions = 0
            test_scores = []
            for itr, (x, y) in enumerate(train_loader):
                #inputs = Variable(x)
                #labels = Variable(y)

                cuda_batch = x.cuda(gpu_device)
                cuda_targets = y.cuda(gpu_device)
                #print('gpu_dev is ', gpu_device)
                inputs = Variable(cuda_batch, requires_grad=False)
                labels = Variable(cuda_targets, requires_grad=False)

                optim.zero_grad()
                pred_labels = model.forward(inputs)
                loss = self.loss_func(pred_labels, labels)
                self.train_loss_history.append(loss.data[0])


                y_pred = np.argmax(pred_labels.data.cpu().numpy(), axis=1)
                # print(y_pred)
                y_actual = labels.data.cpu().numpy()
                y_actual_mask = np.where(y_actual >= 0)
                # print (y_actual)
                pred_acc = np.mean((np.array(y_pred) == np.array(y_actual))[y_actual_mask])
                test_scores.append(pred_acc)

                #inputs, labels = Variable(batch[0]), Variable(batch[1])
                #total_predictions += y_actual.size
                #correct_predictions += pred_acc
                loss.backward()
                optim.step()
                if (itr + 1) % log_nth == 0:
                    print('[Iteration %d/%d] TRAIN loss: %.4f' % (itr + 1, iter_per_epoch, loss.data[0]))

            #train_acc = float(correct_predictions) / total_predictions
            train_acc = np.mean(test_scores)
            self.train_acc_history.append(train_acc)
            print('[Epoch %d/%d] TRAIN acc/loss: %.4f/%.4f' % (epoch + 1, num_epochs, float(train_acc), loss.data[0]))

            total_predictions = 0
            correct_predictions = 0
            test_scores = []
            for itr, (x, y) in enumerate(val_loader):
                #inputs = Variable(x)
                #labels = Variable(y)

                cuda_batch = x.cuda(gpu_device)
                cuda_targets = y.cuda(gpu_device)
                inputs = Variable(cuda_batch, requires_grad=False)
                labels = Variable(cuda_targets, requires_grad=False)

                pred_labels = model.forward(inputs)
                loss = self.loss_func(pred_labels, labels)
                #_, predicted = torch.max(pred_labels.data, 1)
                # correct_predictions += (predicted == labels).sum()

                y_pred = np.argmax(pred_labels.data.cpu().numpy(), axis=1)
                # print(y_pred)
                y_actual = labels.data.cpu().numpy()
                y_actual_mask = np.where(y_actual >= 0)
                # print (y_actual)
                pred_acc = np.mean((np.array(y_pred) == np.array(y_actual))[y_actual_mask])
                test_scores.append(pred_acc)

                #total_predictions += y_actual.size
                #correct_predictions += pred_acc
            #val_acc = float(correct_predictions) / total_predictions
            val_acc = np.mean(test_scores)
            self.val_acc_history.append(val_acc)
            print('[Epoch %d/%d] VAL acc/loss: %.4f/%.4f'
                  % (epoch + 1, num_epochs,float(val_acc), loss.data[0]))

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        print 'FINISH.'
