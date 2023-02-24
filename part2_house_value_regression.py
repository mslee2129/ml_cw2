import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
import pandas as pd
from sklearn import preprocessing

class Regressor():

    def __init__(self, x, nb_epoch = 1000, neurons = [10,10], activations=["relu", "identity"]):
        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests
        """ 
        Initialise the model.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape 
                (batch_size, input_size), used to compute the size 
                of the network.
            - nb_epoch {int} -- number of epochs to train the network.
            - neurons {list} -- Number of neurons in each linear layer 
                represented as a list. The length of the list determines the 
                number of linear layers.
            - activations {list} -- List of the activation functions to apply 
                to the output of each linear layer.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Setting up necessary attributes
        self.x, _ = self._preprocessor(x, training = True)
        self.input_size = self.x.shape[1]
        self.output_size = 1
        self.nb_epoch = nb_epoch 
        self.neurons = neurons
        self.activations = activations

        # Initialize Neural Network
        self.network = Net(self.input_size, self.output_size, self.neurons, self.activations)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def _preprocessor(self, x, y = None, training = False):
        """ 
        Preprocess input of the network.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or 
                testing the model.

        Returns:
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed input array of
              size (batch_size, input_size). The input_size does not have to be the same as the input_size for x above.
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed target array of
              size (batch_size, 1).
            
        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Return preprocessed x and y, return None for y if it was None
        if y != None: # Preprocess y if it is not None
            # normalise y
            min_max_scaler = preprocessing.MinMaxScaler()
            np_y = np.array(y).reshape(-1, 1)
            y = pd.DataFrame(min_max_scaler.fit_transform(np_y))
            # conver to torch tensor
            y = torch.tensor(y.values)
            
        # If training is false, return the preprocessed dataset
        if training == False:
            return self.x, y #return y whether it is None or not

        #################################
        # TRAINING IS TRUE FOR THE BELOW:
        #################################
        # fill empty values with 0
        x.fillna(0)

        # perform one-hot encoding on ocean_proximity
        # categories are: ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN']
        lb = preprocessing.LabelBinarizer()
        lb.fit(x.ocean_proximity)
        # store 1-hot encoded data into binarised_ocean
        binarised_ocean_proximity = lb.transform(x['ocean_proximity'])
        # print(binarised_ocean_proximity)
        # x['ocean_proximity'] = pd.DataFrame([binarised_ocean])
        
        for i in range(len(x)):
            x['ocean_proximity'][i] = binarised_ocean_proximity[i]
        
        # perform constant normalisation on columns
        columns = ['longitude', 'latitude', 'median_income', 'housing_median_age',
                    'total_rooms', 'total_bedrooms', 'population', 'households']

        min_max_scaler = preprocessing.MinMaxScaler()

        for col in columns:
            np_x = np.array(x[col]).reshape(-1, 1)
            x[col] = pd.DataFrame(min_max_scaler.fit_transform(np_x))

        print(x)
        x = torch.tensor(x.values)
        return (x, y)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

        
    def fit(self, x, y):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, Y = self._preprocessor(x, y = y, training = True) # Do not forget
        
        return self

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

            
    def predict(self, x):
        """
        Output the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).

        Returns:
            {np.ndarray} -- Predicted value for the given input (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, _ = self._preprocessor(x, training = False) # Do not forget
        pass

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def score(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, Y = self._preprocessor(x, y = y, training = False) # Do not forget
        return 0 # Replace this code with your own

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


def save_regressor(trained_model): 
    """ 
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open('part2_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")


def load_regressor(): 
    """ 
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open('part2_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model



def RegressorHyperParameterSearch(): 
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented 
    in the Regressor class.

    Arguments:
        Add whatever inputs you need.
        
    Returns:
        The function should return your optimised hyper-parameters. 

    """

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################

    return  # Return the chosen hyper parameters

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################



class Net(nn.Module):
    def __init__(self, input_size, neurons, activations):
        super(Net, self).__init__()

        self._layers = []
        
        for i in range(len(neurons)):
            layer = None
            if(i == 0):
                # First layer is created differently with input_dim
                self._layers.append(nn.Linear(input_size, neurons[i]))
            
            else:
                # All other linear layers are created with num_neurons from previous and next
                layer = nn.Linear(neurons[i-1], neurons[i])
                self._layers.append(layer)

            
            # Create the activation layers to apply to the output of each linear layer
            if activations[i] == "relu":
                self._layers.append(nn.ReLU())

            elif activations[i] == "sigmoid":
                self._layers.append(nn.Sigmoid())

            else:
                self._layers.append(nn.Identity())



    def forward(self, x):
        for layer in self._layers:
            x = layer.forward(x)
        
        return x

    def backward(self, grad_z):
        #create the flow (passes through the layers)
        return grad_z


def example_main():

    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas DataFrame as inputs
    data = pd.read_csv("housing.csv") 

    # Splitting input and output
    x_train = data.loc[:, data.columns != output_label]
    y_train = data.loc[:, [output_label]]

    # Training
    # This example trains on the whole available dataset. 
    # You probably want to separate some held-out data 
    # to make sure the model isn't overfitting
    regressor = Regressor(x_train, nb_epoch = 10)
    regressor.fit(x_train, y_train)
    save_regressor(regressor)

    # # Error
    # error = regressor.score(x_train, y_train)
    # print("\nRegressor error: {}\n".format(error))


if __name__ == "__main__":
    example_main()

