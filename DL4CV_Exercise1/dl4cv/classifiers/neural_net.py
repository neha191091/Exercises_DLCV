import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, IncrementalPCA


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network. The net has an input dimension of
    N, a hidden layer dimension of H, and performs classification over C classes.
    We train the network with a softmax loss function and L2 regularization on the
    weight matrices. The network uses a ReLU nonlinearity after the first fully
    connected layer.
  
    In other words, the network has the following architecture:
  
    input - fully connected layer - ReLU - fully connected layer - softmax
  
    The outputs of the second fully-connected layer are the scores for each class.
    """

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:
    
        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)
    
        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        """
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        # hand-made pca
        self.eigValues = None
        self.eigVectors = None
        self.mean = None

        # sklearn pca
        self.Xpca = None    #saves the transformed training set
        self.pca = None     #saves the pca object

        # normalization
        self.X_train_mean = None
        self.X_train_std = None

    def reinit_params(self, input_size, hidden_size, output_size, std=1e-4):
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def loss(self, X, y=None, reg=0.0):
        """
        Compute the loss and gradients for a two layer fully connected neural
        network.
    
        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
          an integer in the range 0 <= y[i] < C. This parameter is optional; if it
          is not passed then we only return scores, and if it is passed then we
          instead return the loss and gradients.
        - reg: Regularization strength.
    
        Returns:
        If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
        the score for class c on input X[i].
    
        If y is not None, instead return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of training
          samples.
        - grads: Dictionary mapping parameter names to gradients of those parameters
          with respect to the loss function; has the same keys as self.params.
        """
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape
        H, C = W2.shape

        #Our network is : x ---W1,b1---> a1 --- h1 = relu ----> z1 ---W2,b2---> a2 ---h2 = softmax error---> loss

        # Compute the forward pass
        scores = None
        #############################################################################
        # TODO: Perform the forward pass, computing the class scores for the input. #
        # Store the result in the scores variable, which should be an array of      #
        # shape (N, C).                                                             #
        #############################################################################
        #pass
        W1_bias = np.row_stack((W1,b1))   # (D+1,H)
        X_aug = np.column_stack((X,np.ones(X.shape[0]))) #(N,D+1)

        #a1 = X.dot(W1) + b1 # (N,H)

        a1 = X_aug.dot(W1_bias) # (N,H)
        z1 = a1*(a1>0)      # (N,H)

        W2_bias = np.row_stack((W2,b2))   # (H+1,C)
        z1_aug = np.column_stack((z1,np.ones(z1.shape[0]))) # (N, H+1)

        #a2 = z1.dot(W2) + b2 # (N,C)

        a2 = z1_aug.dot(W2_bias)  # (N,C)
        scores = a2
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        # If the targets are not given then jump out, we're done
        if y is None:
            return scores

        # Compute the loss
        loss = None
        #############################################################################
        # TODO: Finish the forward pass, and compute the loss. This should include  #
        # both the data loss and L2 regularization for W1 and W2. Store the result  #
        # in the variable loss, which should be a scalar. Use the Softmax           #
        # classifier loss. So that your results match ours, multiply the            #
        # regularization loss by 0.5                                                #
        #############################################################################
        #pass
        posterior = np.exp(a2) / np.sum(np.exp(a2), axis=1)[:, None] # z2 : (N,C)
        posterior_c = np.choose(y,posterior.T)  # (N)

        #L is actually -np.log(posterior_c)/N, dim: N

        loss = np.sum(-np.log(posterior_c))/N + 0.5*reg*(np.sum(W1_bias*W1_bias) + np.sum(W2_bias*W2_bias)) #can add bias regularization
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        # Backward pass: compute gradients
        grads = {}


        #############################################################################
        # TODO: Compute the backward pass, computing the derivatives of the weights #
        # and biases. Store the results in the grads dictionary. For example,       #
        # grads['W1'] should store the gradient on W1, and be a matrix of same size #
        #############################################################################
        #pass


        #For W2 and b2

        # convert the labels to one-hot encoding
        yPos = np.zeros_like(posterior)
        yPos[np.arange(N), y] = 1  # testa = np.zeros((4,4)),  testa[[2,3],[3,2]] = 1
        # print(yPos)

        # gradient
        dLa2 = (posterior - yPos) #Loss is the log of correct posterior, dim: (N,C)
        dW2_bias = z1_aug.T.dot(dLa2) # (H+1,C), remember z1_aug was (N,H+1) - basically all the gradients for all considered data are
                             # added up -> our loss is a sum (avg really) of the loss for each data point

        # regularization
        # dW1 = dW1 + reg * (W1 + W2)

        # avg gradient
        dW2_bias = dW2_bias / N

        # regularization
        dW2_bias = dW2_bias + reg * W2_bias

        # split into gradient for weight and bias
        grads['W2'] = dW2_bias[0:H,:]
        grads['b2'] = dW2_bias[H]



        #For W1 and b1

        a1criterion = (a1 >= 0) # (N,H)
        #print('a1criterion', a1criterion)

        dLa2_W2_prod = dLa2.dot(W2.T) # (N,C)x(C,H)=(N,H)

        dW1_bias = X_aug.T.dot(dLa2_W2_prod*a1criterion)/N + reg * W1_bias

        #split into gradient for weight and bias
        grads['W1'] = dW1_bias[0:D,:]
        grads['b1'] = dW1_bias[D]






        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return loss, grads

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=1e-5, num_iters=100,
              batch_size=200, verbose=False, trainafresh = True):
        """
        Train this neural network using stochastic gradient descent.
    
        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
          X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning rate
          after each epoch.
        - reg: Scalar giving regularization strength.
        - num_iters: Number of steps to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        """

        if(trainafresh == True or self.pca == None or self.Xpca == None):
            #eigValues, eigVectors, mean = self.eigAnalysis(X)
            #X_pca = self.scaledFeaturesVectors(X, eigVectors, eigValues, mean)
            #X_pca_val = self.scaledFeaturesVectors(X_val, eigVectors, eigValues, mean)
            #self.eigValues = eigValues
            #self.eigVectors = eigVectors
            #self.mean = mean

            X_pca, pca = self.pca_sklearn(X,X.shape[1],ica=True)

            self.Xpca = X_pca
            self.pca = pca
        else:
            X_pca = self.Xpca

        print("data shape X_pca; ", X_pca.shape)
        #print("XPCA: ", X_pca)
        print("data shape x: ", X.shape)
        #print("X: ", X)
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)

        print('iterations_per_epoch', iterations_per_epoch)

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in xrange(num_iters):
            X_batch = None
            y_batch = None

            #########################################################################
            # TODO: Create a random minibatch of training data and labels, storing  #
            # them in X_batch and y_batch respectively.                             #
            #########################################################################
            #pass
            mask = np.random.choice(num_train, batch_size, replace=True)
            X_batch = X_pca[mask] #X_batch = X[mask]
            y_batch = y[mask]

            #########################################################################
            #                             END OF YOUR CODE                          #
            #########################################################################

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            #########################################################################
            # TODO: Use the gradients in the grads dictionary to update the         #
            # parameters of the network (stored in the dictionary self.params)      #
            # using stochastic gradient descent. You'll need to use the gradients   #
            # stored in the grads dictionary defined above.                         #
            #########################################################################
            #pass
            #update W2
            self.params['W2'] -= learning_rate * grads['W2']
            self.params['b2'] -= learning_rate * grads['b2']
            self.params['W1'] -= learning_rate * grads['W1']
            self.params['b1'] -= learning_rate * grads['b1']

            #########################################################################
            #                             END OF YOUR CODE                          #
            #########################################################################

            if verbose and it % 100 == 0:
                print 'iteration %d / %d: loss %f' % (it, num_iters, loss)

            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # Decay learning rate
                learning_rate *= learning_rate_decay
                print('learning rate on iteration: ', it, " is: ", learning_rate )
                print('loss: ', loss)
                print('train acc: ', train_acc)
                print('val acc: ', val_acc)

        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        """
        Use the trained weights of this two-layer network to predict labels for
        data points. For each data point we predict scores for each of the C
        classes, and assign each data point to the class with the highest score.
    
        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
          classify.
    
        Returns:
        - y_pred: A numpy array of shape (N,) giving predicted labels for each of
          the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
          to have class c, where 0 <= c < C.
        """
        y_pred = None

        #pre-processing with pca

        #X_pca = self.scaledFeaturesVectors(X, self.eigVectors, self.eigValues, self.mean)
        X_pca = self.pca_sklearn_transform(X)

        ###########################################################################
        # TODO: Implement this function; it should be VERY simple!                #
        ###########################################################################
        #pass

        layer1_in = X_pca.dot(self.params['W1']) + self.params['b1']
        layer1_out = layer1_in*(layer1_in>0)
        layer2_in = layer1_out.dot(self.params['W2']) + self.params['b2']
        y_pred = np.argmax(layer2_in, axis=1)

        ###########################################################################
        #                              END OF YOUR CODE                           #
        ###########################################################################

        return y_pred

    def normalize(self, X):
        if(self.x_train_mean == None or self.X_train_std == None):
            self.X_train_mean = np.mean(X, axis=0, keepdims=True)
            X -= self.X_train_mean
            self.X_train_std = np.std(X, axis=0, keepdims=True)
            X /= self.X_train_std
        else:
            X -= self.X_train_mean
            X /= self.X_train_std


    def pca_sklearn(self,X,n_components,whiten=False,ica=False,icaBatchSize=-1, normalize=False):
        if(ica == True):
            if(icaBatchSize != -1):
                pca = IncrementalPCA(n_components, whiten, batch_size=icaBatchSize)
            else:
                pca = IncrementalPCA(n_components, whiten)
        else:
            pca = PCA(n_components, True, whiten)
        if(normalize == True):
            X = self.normalize(X)
        X_scaled = pca.fit_transform(X)
        return X_scaled, pca

    def pca_sklearn_transform(self,X,normalize=False):
        if (normalize == True):
            X = self.normalize(X)
        return self.pca.transform(X)

    def scaledFeaturesVectors(self, dataFeatures, eigVectors, eigValues, mean):
        sc = []
        dim = np.shape(dataFeatures)
        c = np.matmul(eigValues, np.transpose(eigVectors))
        for i in range(dim[0]):
            m = dataFeatures[i, :] - mean
            sc.append(np.matmul(c, m))
        return (np.array(sc))

    def eigAnalysis(self, dataFeatures_train):
        mean = np.mean(dataFeatures_train,axis=0)
        # dataFeatures_train = minuMean(dataFeatures_train)
        cov = np.cov(dataFeatures_train.T)
        eigValues, eigVectors = np.linalg.eig(cov)
        for i in range(eigValues.size):
            if eigValues[i] < 0:
                # print(eigValues[i])
                eigValues[i] = 0.000000001
        eigValuesD = np.diag(eigValues)
        eigValuesD = np.linalg.inv(eigValuesD)
        eigValuesD = np.sqrt(eigValuesD)
        return eigValuesD, eigVectors, mean

    def test(self):
        print ('testing')

