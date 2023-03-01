import pickle
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, r2_score
import part1_nn_lib as nn


class Regressor():

    def __init__(self, x, nb_epoch = 100, neurons = [5,1], activations=["sigmoid", "identity"], batch_size = 8000, learning_rate=0.01, shuffle_flag=True):
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
        # Preprocessing values
        self.minValuesX = []
        self.maxValuesX = []
        self.minValuesY = []
        self.maxValuesY = []
        self.upperBound = 1.0
        self.lowerBound = 0.0
        self.x, _ = self._preprocessor(x, training = True)
        
        # Setting up necessary attributes
        self.input_size = self.x.shape[1]
        self.nb_epoch = nb_epoch 
        self.neurons = neurons
        self.activations = activations
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.shuffle_flag= shuffle_flag
        
        # Initialize Neural Network
        net = nn.MultiLayerNetwork(self.input_size, self.neurons, self.activations)

        # Initialize Trainer
        self.network = nn.Trainer(
            network=net,
            batch_size=self.batch_size,
            nb_epoch=self.nb_epoch,
            learning_rate=self.learning_rate,
            loss_fun="mse",
            shuffle_flag=True,
        )

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

        # PREPROCESSING Y
        if y is not None:
            # Calculate preprocessing values of Y, if not None
            y = (np.array(y)).astype(float)
            self.minValuesY = np.min(y, axis=0)
            self.maxValuesY = np.max(y, axis=0)
            y = self.lowerBound + ((y - self.minValuesY) * (self.upperBound - self.lowerBound) / (self.maxValuesY - self.minValuesY))
        
        # if y is not None:
        #     # Check if some Y values are missing, if so remove the whole line
        #     # Get the index values of all the Y rows with empty values
        #     drop_index = y[y.isna()].index.tolist()

        #     # Drop them from Y
        #     y = y.drop(drop_index, axis = 0)
        #     # Drop them from X
        #     x = x.drop(drop_index, axis = 0)


        # REMOVING NA VALUES FROM X
        x = x.fillna(0)

# TO DO, DO NOT HARD CODE THE NAME OF THE COLUMNS BELOW

        # DEALING WITH CATEGORICAL VARIABLES
        binarised_ocean_proximity = pd.DataFrame(
            preprocessing.label_binarize(
                x.ocean_proximity, 
                classes=['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'],  # keeping the order of the columns constant
            ),
            columns= ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'] # adding column names
        ) 
        
        x = x.drop(labels='ocean_proximity', axis=1) #deleting categorical column before adding the new one
        x.reset_index(drop=True, inplace=True)
        x = pd.concat([x, binarised_ocean_proximity], axis=1) # adding the 5 dummy columns
            
        # IF TESTING, USE STORED PREPROCESSED ATTRIBUTES FOR X
        if not training:
            x = (np.array(x)).astype(float)
            x = self.lowerBound + ((x - self.minValuesX) * (self.upperBound - self.lowerBound) / (self.maxValuesX - self.minValuesX))

            return (x, y)

        # IF TRAINING
        # PREPROCESSING X
        x = (np.array(x)).astype(float)
        self.minValuesX = np.min(x, axis=0)
        self.maxValuesX = np.max(x, axis=0)
        x = self.lowerBound + ((x - self.minValuesX) * (self.upperBound - self.lowerBound) / (self.maxValuesX - self.minValuesX))

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
        X, Y = self._preprocessor(x, y, training = True)
        self.network.train(X, Y)

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
        X, _ = self._preprocessor(x, training = False)
        print("X after preprocessing unnormalise: ", X)
        norm_pred = self.network.network(X).squeeze()
        print("pre unnormalise: ", norm_pred)
        
        predictedValues = self.minValuesY + ((norm_pred - self.lowerBound) * (self.maxValuesY - self.minValuesY)) / (self.upperBound - self.lowerBound)
        print(predictedValues)
        return predictedValues

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
        y = (np.array(y)).astype(float).squeeze()
        predictions = self.predict(x)
        print (y)
        rmse = np.sqrt(mean_squared_error(y, predictions))


        print("\n|------------- MODEL PERFORMANCE -------------|")
        print("|            root_mean_squared_error                 |")
        print("|                 ",rmse,"                       |")
        # print("|                   PRECISION                 |")
        # print("|                    ",precision,"                    |")
        # print("|                   RECALL                    |")
        # print("|                    ",recall,"                    |")
        print("|---------------------------------------------|\n")
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


def example_main():

    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas DataFrame as inputs
    data = pd.read_csv("housing.csv") 
    data = data.sample(frac=1).reset_index(drop=True)


    # Splitting input and output
    x= data.loc[:, data.columns != output_label]
    y = data.loc[:, [output_label]]

    split_idx = int(0.8 * len(x))
    x_train = x[:split_idx]
    y_train = y[:split_idx]
    x_test = x[split_idx:]
    y_test = y[split_idx:]

    # print(x_test.shape)
    # print(y_test.shape)
    # Training
    # This example trains on the whole available dataset. 
    # You probably want to separate some held-out data 
    # to make sure the model isn't overfitting
    regressor = Regressor(x_train)
    regressor.fit(x_train, y_train)
    save_regressor(regressor)

    reg = load_regressor()

    # # Error
    error = reg.score(x_test,y_test)
    print("\nRegressor error: {}\n".format(error))


if __name__ == "__main__":
    example_main()
