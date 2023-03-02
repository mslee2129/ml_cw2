import pickle
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, r2_score
import part1_nn_lib as nn
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator
import matplotlib.pyplot as plt


class Regressor(BaseEstimator):

    def __init__(self, x, nb_epoch = 100, neurons = [20,20,1], activations=["sigmoid", "sigmoid", "identity"], batch_size = 32, learning_rate=0.01, shuffle_flag=True, dropout_rate=0):
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
        # self.x, _ = self._preprocessor(x, training = True)
        self.x = x
        # Setting up necessary attributes
        self.input_size = self.x.shape[1]+4
        self.nb_epoch = nb_epoch 
        self.neurons = neurons
        self.activations = activations
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.shuffle_flag= shuffle_flag
        self.dropout_rate = dropout_rate
        
        # Initialize Neural Network
        net = nn.MultiLayerNetwork(self.input_size, self.neurons, self.activations, dropout_rate=self.dropout_rate)

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

        #REPORT
        # REMOVING NA VALUES FROM X 
        # print("Mean of total bedrooms", sum(x['total_bedrooms'] == x['total_bedrooms'].mean()))
        #print(x.isna().sum()) 
        x = x.fillna(x.mean())
        # print(x.isna().sum()) 
        #print("Mean of total bedrooms", sum(x['total_bedrooms'] == x['total_bedrooms'].mean()))

        # IF TESTING, USE STORED PREPROCESSED ATTRIBUTES FOR X
        if not training:
            x = (np.array(x)).astype(float)
            x[:,:-5] = self.lowerBound + ((x[:,:-5] - self.minValuesX) * (self.upperBound - self.lowerBound) / (self.maxValuesX - self.minValuesX))

            return (x, y)

        # IF TRAINING
        # PREPROCESSING X
        x = (np.array(x)).astype(float)
        self.minValuesX = np.min(x[:,:-5], axis=0)
        self.maxValuesX = np.max(x[:,:-5], axis=0)
        x[:,:-5] = self.lowerBound + ((x[:,:-5] - self.minValuesX) * (self.upperBound - self.lowerBound) / (self.maxValuesX - self.minValuesX))

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
        # print("\n X after preprocessing", X[-100:,-5:])
        # print("Same as above, this time number of NA:", np.sum(np.isnan(X)))

        norm_pred = self.network.network(X, True).squeeze()
        # print("\n pre unnormalise: ", norm_pred)
        
        predictedValues = self.minValuesY + ((norm_pred - self.lowerBound) * (self.maxValuesY - self.minValuesY)) / (self.upperBound - self.lowerBound)
        # print("\n Predicted Values once unnormalised", predictedValues)
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
        # print (y)
        rmse = np.sqrt(mean_squared_error(y, predictions))


        # print("\n|------------- MODEL PERFORMANCE -------------|")
        # print("|  root_mean_squared_error                 ")
        # print("|  ",rmse)
        # print("|  Average Value of Predictions   ")
        # print("|  ",np.average(predictions))
        # print("| REAL average  ")
        # print("|  ",np.average(y))
        # print("|  Max - Min of Predictions  ")
        # print("|  Max : ",np.max(predictions),"  Min: ", np.min(predictions))
        # print("|  Max - Min of REAL  ")
        # print("|  Max : ",np.max(y),"  Min: ", np.min(y))
    
        # print("|---------------------------------------------|\n")
        return rmse
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



def RegressorHyperParameterSearch(x,y): 
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented 
    in the Regressor class.

    Arguments:
        - x {pd.DataFrame} -- Raw input array of shape 
                (dataset_size, input_size).
        - y {pd.DataFrame} -- Raw output array of shape (dataset_size, 1).
    Returns:
        The function should return your optimised hyper-parameters. 

    """

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################

    # nb_epoch = [100, 500, 1000]
    # neurons = [[20,10,5,1],[30,30,15,1]]
    # batch_size = [32, 64, 128, 256]
    # learning_rate = [0.01, 0.05]
    # activations = [["relu","relu","relu", "identity"],["sigmoid", "sigmoid", "sigmoid", "identity"]]

    nb_epoch = [50]
    neurons = [[20,10,5,1]]
    batch_size = [32]
    learning_rate = [0.01]
    activations = [["relu","relu","relu", "identity"]]
    dropout_rate = [0.1]
    
    param_grid = {
        "nb_epoch" : nb_epoch,
        "neurons" : neurons,
        "batch_size": batch_size,
        "learning_rate" : learning_rate,
        "activations" : activations,
        "dropout_rate" : dropout_rate
    }
    
    regressor = Regressor(x)
    
    optimal_model = GridSearchCV(
        estimator=regressor,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        verbose=4
        ).fit(x,y)

    with open("result.pickle", "wb") as target:
        pickle.dump(optimal_model, target)
        
    # print(optimal_model.best_estimator_ )
    print(type(optimal_model.best_estimator_))



    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################




def overfitting_analysis():
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

    epochs = []
    no_dropout_test_errors = []
    no_dropout_eval_errors = []

    dropout_test_errors = []
    dropout_eval_errors = []

    for epoch in range(100, 2001, 100):
        print("Currently at epoch:", epoch)
        epochs.append(epoch)

        no_dropout_regressor = Regressor(x_train, nb_epoch=epoch, dropout_rate=0)
        no_dropout_regressor.fit(x_train, y_train)
        save_regressor(no_dropout_regressor)
        ndr = load_regressor()
        no_dropout_test_errors.append(ndr.score(x_test,y_test))
        no_dropout_eval_errors.append(ndr.score(x_train,y_train))
        
        
        dropout_regressor = Regressor(x_train, nb_epoch=epoch, dropout_rate=0.2)
        dropout_regressor.fit(x_train, y_train)
        save_regressor(dropout_regressor)
        dr = load_regressor()
        dropout_test_errors.append(dr.score(x_test,y_test))
        dropout_eval_errors.append(dr.score(x_train,y_train))

    return (epochs, dropout_eval_errors, no_dropout_eval_errors, dropout_test_errors, no_dropout_test_errors)


def graph_it(epochs, dropout_eval_errors, no_dropout_eval_errors, dropout_test_errors, no_dropout_test_errors):
    plt.figure(figsize=(8,6))
    plt.plot(epochs, dropout_eval_errors, label='Eval RMSE')
    plt.plot(epochs, dropout_test_errors, label='Test RMSE')
    plt.title("RMSE per epoch for eval")
    plt.xlabel("Epochs")
    plt.ylabel("RMSE Loss")
    plt.legend()
    file_name = "graphs/Eval"
    plt.savefig(file_name)
    plt.clf() # Clears the figure so the graphs don't overlap in the saved file

    plt.figure(figsize=(8,6))
    plt.plot(epochs, no_dropout_eval_errors, label='Eval RMSE')
    plt.plot(epochs, no_dropout_test_errors, label='Test RMSE')
    plt.title("RMSE per epoch for test")
    plt.xlabel("Epochs")
    plt.ylabel("RMSE Loss")
    plt.legend()
    file_name = "graphs/Test"
    plt.savefig(file_name)
    plt.clf() # Clears the figure so the graphs don't overlap in the saved file

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

    # epochs, dropout_eval_errors, no_dropout_eval_errors, dropout_test_errors, no_dropout_test_errors = overfitting_analysis()

    # print("no_dropout_eval_errors", no_dropout_eval_errors)
    # print("dropout_eval_errors",dropout_eval_errors)
    
    # print("no_dropout_test_errors", no_dropout_test_errors)
    # print("dropout_test_errors", dropout_test_errors)
    

    # graph_it(epochs, dropout_eval_errors, no_dropout_eval_errors, dropout_test_errors, no_dropout_test_errors)
