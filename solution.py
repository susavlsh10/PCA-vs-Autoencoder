import torch
import torch.nn as nn
import numpy as np
import pickle, tqdm, os, time


class PCA():
    def __init__(self, X, n_components):
        '''
        Args:
            X: The data matrix of shape [n_features, n_samples].
            n_components: The number of principal components. A scaler number.
        '''

        self.n_components = n_components
        self.X = X
        self.Up, self.Xp = self._do_pca()

    
    def _do_pca(self):
        '''
        To do PCA decomposition.
        Returns:
            Up: Principal components (transform matrix) of shape [n_features, n_components].
            Xp: The reduced data matrix after PCA of shape [n_components, n_samples].
        '''
        ### YOUR CODE HERE
        n=self.X.shape[1]
        _X= (1/n) * np.dot(self.X, np.ones(n))
        one_n= np.ones(n)
        
        X_mean= np.outer(_X, one_n)
        X_hat= self.X - X_mean  #zero mean 
        
        u, s, v = np.linalg.svd(X_hat) #Singular value decomposition
        
        Up = u[:, 0: self.n_components]
        
        Xp = np.dot(Up.T, self.X)
        
        ### END YOUR CODE
        return Up, Xp

    def get_reduced(self, X=None):
        '''
        To return the reduced data matrix.
        Args:
            X: The data matrix with shape [n_features, n_any] or None. 
               If None, return reduced training X.
        Returns:
            Xp: The reduced data matrix of shape [n_components, n_any].
        '''
        if X is None:
            return self.Xp, self.Up
        else:
            return self.Up.T @ X, self.Up

    def reconstruction(self, Xp):
        '''
        To reconstruct reduced data given principal components Up.

        Args:
        Xp: The reduced data matrix after PCA of shape [n_components, n_samples].

        Return:
        X_re: The reconstructed matrix of shape [n_features, n_samples].
        '''
        ### YOUR CODE HERE
        
        X_re = np.dot(self.Up, Xp) 
        
        ### END YOUR CODE
        return X_re


def frobeniu_norm_error(A, B):
    '''
    To compute Frobenius norm's square of the matrix A-B. It can serve as the
    reconstruction error between A and B, or can be used to compute the 
    difference between A and B.

    Args: 
    A & B: Two matrices needed to be compared with. Should be of same shape.

    Return: 
    error: the Frobenius norm's square of the matrix A-B. A scaler number.
    '''
    return np.linalg.norm(A-B)


class AE(nn.Module): 
    def __init__(self, d_hidden_rep):
        '''
        Args:
            d_hidden_rep: The dimension for the hidden representation in AE. A scaler number.
            n_features: The number of initial features, 256 for this dataset.
            
        Attributes:
            X: A torch tensor of shape [256, None]. A placeholder 
               for input images. "None" refers to any batch size.
            out_layer: A torch tensor of shape [256, None]. Output signal
               of network
            initializer: Initialize the trainable weights.
        '''
        super(AE, self).__init__()
        self.d_hidden_rep = d_hidden_rep
        self.n_features = 256
        self._network()
        
    def _network(self):

        '''
        Note: you should include all the three variants of the networks here. 
        You can comment the other two when you running one, but please include 
        and uncomment all the three in you final submissions.
        '''

        # Note: here for the network with weights sharing. Basically you need to follow the
        #---------------------network with and without weights sharing------------------------------------------------------#
        
        self.w = torch.empty(self.d_hidden_rep, self.n_features)
        nn.init.kaiming_uniform_(self.w, mode='fan_in', nonlinearity='relu')
        
        self.encode= nn.Linear(self.n_features, self.d_hidden_rep, bias=False)
        self.encode.weight= nn.Parameter(self.w)
        self.decode= nn.Linear(self.d_hidden_rep, self.n_features, bias=False)
        
        # Note: here for the network without weights sharing 
        
        #the same code for network with weight sharing will work 


        #-----------------------------------------------------------------------------------------#

        # Note: here for the network with more layers and nonlinear functions  
        
        self.Encode= nn.Sequential(
            nn.Linear(in_features=256, out_features=200),
            nn.ReLU(),
            nn.Linear(in_features=200, out_features=150),
            nn.ReLU(),
            nn.Linear(in_features=150, out_features=self.d_hidden_rep),
            nn.ReLU(),            
            
            )
        
        self.Decode= nn.Sequential(
            nn.Linear(in_features=self.d_hidden_rep, out_features=150),
            nn.ReLU(),
            nn.Linear(in_features=150, out_features=200),
            nn.ReLU(),
            nn.Linear(in_features=200, out_features=256),
            nn.Sigmoid(),            
            )  
        
        

        
        ### END YOUR CODE
    
    def _forward(self, X):
        '''

        You are free to use the listed functions and APIs from torch and torch.nn:
            torch.mm
            torch.transpose
            nn.Tanh
            nn.ReLU
            nn.Sigmoid
        
        Args:
            X: A torch tensor of shape [n_features, batch_size].
                for input images.

        Returns:
            out: A torch tensor of shape [n_features, batch_size].
            
        '''

        ### YOUR CODE HERE

        '''
        Note: you should include all the three variants of the networks here. 
        You can comment the other two when you running one, but please include 
        and uncomment all the three in you final submissions.
        '''

        # Note: here for the network with weights sharing. Basically you need to follow the
        # formula (WW^TX) in the note at http://people.tamu.edu/~sji/classes/PCA.pdf
        
        #-------Network with Weights sharing ---------------------#
        
        W=self.encode.weight
        W_T= torch.transpose(W, 0, 1)
        if len(X.shape)==1:
            X=X.view(self.n_features,1)
        out = torch.mm(W, X)
        
        out= torch.mm(W_T, out)
        self.w= W
        #---------------------------------------------------------------#
        
        # Note: here for the network without weights sharing 
        #------Network without weight sharing--------------------#
        
        # W=self.encode.weight
        # DW= self.decode.weight 
        
        # if len(X.shape)==1:
        #     X=X.view(self.n_features,1)
        # out = torch.mm(W, X)
        
        # out= torch.mm(DW, out)
        # self.w= W
       #----------------------------------------------------------------# 

        # Note: here for the network with more layers and nonlinear functions  
       #----------- Network with more layers and nonlinear functions -------#
       
        # if len(X.shape)==1:
        #     X=X.view(self.n_features,1)        
        # X_T= torch.transpose(X, 0, 1)
        # out = self.Encode(X_T)
        # out= self.Decode(out)
        # out= torch.transpose(out, 0, 1)
        
        
        #------------------------------------------------------------------#
        
        return out
        ### END YOUR CODE

    def _setup(self):
        '''
        Model and training setup.
 
        Attributes:
            loss: MSE loss function for computing on the current batch.
            optimizer: torch.optim. The optimizer for training
                the model. Different optimizers use different gradient
                descend policies.
        '''
        self.loss = nn.MSELoss(reduction='mean')
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    
    def train(self, x_train, x_valid, batch_size, max_epoch):

        '''
        Autoencoder is an unsupervised learning method. To compare with PCA,
        it's ok to use the whole training data for validation and reconstruction.
        '''
 
        self._setup()
 
        num_samples = x_train.shape[1]
        num_batches = int(num_samples / batch_size)
 
        num_valid_samples = x_valid.shape[1]
        num_valid_batches = (num_valid_samples - 1) // batch_size + 1

        print('---Run...')
        for epoch in range(1, max_epoch + 1):
 
            # To shuffle the data at the beginning of each epoch.
            shuffle_index = np.random.permutation(num_samples)
            curr_x_train = x_train[:, shuffle_index]
 
            # To start training at current epoch.
            loss_value = []
            qbar = tqdm.tqdm(range(num_batches))
            for i in qbar:
                batch_start_time = time.time()
 
                start = batch_size * i
                end = batch_size * (i + 1)
                x_batch = curr_x_train[:, start:end]

                x_batch_tensor = torch.tensor(x_batch).float()
                x_batch_re_tensor = self._forward(x_batch_tensor)
                loss = self.loss(x_batch_re_tensor, x_batch_tensor)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if not i % 10:
                    qbar.set_description(
                        'Epoch {:d} Loss {:.6f}'.format(
                            epoch, loss.detach().item()))
 
            # To start validation at the end of each epoch.
            loss = 0
            print('Doing validation...', end=' ')
            
            with torch.no_grad():
                for i in range(num_valid_batches):
                    start = batch_size * i
                    end = min(batch_size * (i + 1), x_valid.shape[1])
                    x_valid_batch = x_valid[:, start:end]
    
                    x_batch_tensor = torch.tensor(x_valid_batch).float()
                    x_batch_re_tensor = self._forward(x_batch_tensor)
                    loss = self.loss(x_batch_re_tensor, x_batch_tensor)
 
            print('Validation Loss {:.6f}'.format(loss.detach().item()))
 

    def get_params(self):
        """
        Get parameters for the trained model.
        
        Returns:
            final_w: A numpy array of shape [n_features, d_hidden_rep].
        """
        # try:
        #     self.w= self.encode.weight
        # except AttributeError:
        #     self.w=torch.ones(1)
        return self.w.detach().numpy().T
    
    def reconstruction(self, X):
        '''
        To reconstruct data. You’re required to reconstruct one by one here,
        that is to say, for one loop, input to the network is of the shape [n_features, 1].
        Args:
            X: The data matrix with shape [n_features, n_any], a numpy array.
        Returns:
            X_re: The reconstructed data matrix, which has the same shape as X, a numpy array.
        '''
        _, n_samples = X.shape
        X_re= np.zeros(X.shape)
        with torch.no_grad():
            for i in range(n_samples):
                ### YOUR CODE HERE
                curr_X= X[:, i]
                # Note: Format input curr_X to the shape [n_features, 1]
    
                ### END YOUR CODE            
                curr_X_tensor = torch.tensor(curr_X).float()
                curr_X_re_tensor = self._forward(curr_X_tensor)
                ### YOUR CODE HERE
                X_re[:,i]=  curr_X_re_tensor.view(self.n_features)
                # Note: To achieve final reconstructed data matrix with the shape [n_features, n_any].
            ### END YOUR CODE 
        return X_re
