import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)
  
    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.
  
    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength
  
    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    #pass

    # Gives us the predicted outputs
    output = X.dot(W)
    #print(output.shape)

    # Gives us the list of predicted posterior probability values for each of the data
    # elements for the associated true class
    partProb = np.zeros(X.shape[0])
    sumPartProb = np.zeros(X.shape[0])

    noOfTrainSet = output.shape[0]
    #print(partProb.shape)
    totalloss = 0
    for i in range(0,output.shape[0]):
        #sum = 0
        for j in range(0, output.shape[1]):
            sumPartProb[i] = sumPartProb[i] + np.exp(output[i,j])
        partProb[i] = np.exp(output[i,y[i]])
        partProb[i] = partProb[i]/sumPartProb[i]
        #print(partProb[i])
        totalloss = totalloss - np.log(partProb[i])

    #regularization
    #totalloss = totalloss + 0.5*reg*np.sum(W*W)

    #Get average loss
    loss = totalloss/noOfTrainSet

    #regularization
    totalloss = totalloss + 0.5 * reg * np.sum(W * W)

    #dW_Trans should be (C,D)
    dW_Trans = dW.T
    #print('shape dW', dW.shape)
    #print('shape dW_Trans', dW_Trans.shape)

    #derivative with respect to weight
    for i in range(0, dW_Trans.shape[0]):
        deriv = np.zeros(dW_Trans.shape[1])
        #print('derive shape',deriv.shape)
        for j in range(0, noOfTrainSet):
            if(i == y[j]):
                deriv = deriv - (1 - np.exp(output[j,i]) / (sumPartProb[j])) * X[j]
            else:
                deriv = deriv - (-np.exp(output[j,i])/(sumPartProb[j]))*X[j]
        dW_Trans[i] = deriv
    dW = dW_Trans.T

    #regularization
    #dW = dW + reg*W

    #avg gradient
    dW = dW/noOfTrainSet

    # regularization
    dW = dW + reg * W


    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.
  
    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    noOfTrainSet = X.shape[0]

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    #pass
    # Gives us the predicted outputs (N,C)
    output = X.dot(W)


    # Normalize the output matrix
    #print(np.max(output))
    #print(np.max(output,axis=1))
    #outputmaxMatrix = (np.max(output,axis=1).repeat(output.shape[1])) #Get the maximum of each class
    #outputmaxMatrix = outputmaxMatrix.reshape(output.shape) #stack the column vectors C times
    #print(outputmaxMatrix)
    #output = output - outputmaxMatrix
    #output = output - np.max(output)

    #outputs for the correct classes
    output_c = np.choose(y,output.T)
    #print('correct op shape',output_c.shape)

    #print('sum of output: ' ,np.sum(np.exp(output),axis=1))

    #correct posterior
    posterior_c = np.exp(output_c)/np.sum(np.exp(output),axis=1)
    #print(' posterior', posterior_c)

    loss = - np.sum(np.log(posterior_c))

    # regularization
    #loss = loss + 0.5 * reg * np.sum(W * W)

    #avg loss
    loss = loss/noOfTrainSet

    # regularization
    loss = loss + 0.5 * reg * np.sum(W * W)

    #all posteriors
    posterior = np.exp(output) / np.sum(np.exp(output), axis=1)[:, None]

    #convert the labels to one-hot encoding
    yPos = np.zeros_like(posterior)
    yPos[np.arange(noOfTrainSet),y] = 1 #testa = np.zeros((4,4)),  testa[[2,3],[3,2]] = 1
    #print(yPos)

    # gradient
    dW = X.T.dot(posterior - yPos)

    # regularization
    #dW = dW + reg * W

    #avg gradient
    dW = dW/noOfTrainSet

    # regularization
    dW = dW + reg * W

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW

