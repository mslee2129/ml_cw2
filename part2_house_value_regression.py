import pickle
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
import part1_nn_lib as nn
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator
import matplotlib.pyplot as plt

class Regressor(BaseEstimator):
    
    def __init__(self, 
                 x, 
                 nb_epoch = 100, 
                 neurons = [20,20,1], 
                 activations = ["sigmoid", "sigmoid", "identity"], 
                 batch_size = 32, 
                 learning_rate = 0.01, 
                 shuffle_flag = True, 
                 dropout_rate = 0, 
                 loss_fun = "mse", 
                 upperBound = 1.0, 
                 lowerBound = 0.0):

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
        # Setting attributes
        self.x = x
        self.activations = activations
        self.nb_epoch = nb_epoch
        self.neurons = neurons 
        self.activations = activations
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.shuffle_flag = shuffle_flag
        self.dropout_rate = dropout_rate
        self.loss_fun = loss_fun
        
        # Preprocessing attributes
        self.minValuesX = []
        self.maxValuesX = []
        self.minValuesY = []
        self.maxValuesY = []
        self.upperBound = upperBound
        self.lowerBound = lowerBound

        # Preprocessing data for dimensions
        pre_x = x
        pre_x, _= self._preprocessor(pre_x, training = True)
        self.input_size = pre_x.shape[1]

        # Initialize Neural Network
        self.net = nn.MultiLayerNetwork(self.input_size, self.neurons, 
                                        self.activations, self.dropout_rate)

        # Initialize Trainer
        self.network = nn.Trainer(
            network = self.net,
            batch_size = self.batch_size,
            nb_epoch = self.nb_epoch,
            learning_rate = self.learning_rate,
            loss_fun = self.loss_fun,
            shuffle_flag = self.shuffle_flag,
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
            # Check if some Y values are missing, if so remove the whole line
            drop_index = y[y['median_house_value'].isna()].index.tolist()
            y = y.drop(index=drop_index)
            x = x.drop(index=drop_index)

            # Calculate preprocessing values of Y, if not None
            y = (np.array(y)).astype(float)
            self.minValuesY = np.min(y, axis=0)
            self.maxValuesY = np.max(y, axis=0)
            y = self.lowerBound + ((y - self.minValuesY) * (self.upperBound - self.lowerBound) / (self.maxValuesY - self.minValuesY))
        
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

        x = x.fillna(x.mean())

        # IF TESTING, USE STORED PREPROCESSED ATTRIBUTES FOR X
        if not training:
            x = (np.array(x)).astype(float)
            x[:,:-5] = self.lowerBound + ((x[:,:-5] - self.minValuesX) * (self.upperBound - self.lowerBound) / (self.maxValuesX - self.minValuesX))

            return (x, y)

        # IF TRAINING PREPROCESS X
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

        norm_pred = self.network.network(X).squeeze()
        predictedValues = self.minValuesY + ((norm_pred - self.lowerBound) * (self.maxValuesY - self.minValuesY)) / (self.upperBound - self.lowerBound)

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
    # print("\nLoaded model in part2_model.pickle\n")
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

    # Values to test
    nb_epoch = [50,100,500,1000,5000]
    neurons = [[20,10,5,1],[30,30,15,1],[50,50,25,1]]
    batch_size = [32,64,128]
    learning_rate = [0.01,0.05]
    activations = [["relu", "relu", "relu", "identity"], 
                   ["sigmoid", "sigmoid", "sigmoid", "identity"],
                   ["relu", "sigmoid", "relu", "identity"],
                   ["leakyrelu", "leakyrelu","leakyrelu"]
                   ]
    dropout_rate = [0, 0.1, 0.2, 0.5]

    parameters = {
        "nb_epoch" : nb_epoch,
        "neurons" : neurons,
        "batch_size": batch_size,
        "learning_rate" : learning_rate,
        "activations" : activations,
        "dropout_rate" : dropout_rate
    }
    
    regressor = Regressor(x)
    
    gs = GridSearchCV(
        estimator=regressor,
        param_grid=parameters,
        scoring="neg_root_mean_squared_error",
        verbose=4,
        return_train_score = True,
        nb_jobs = 4
        )
    
    result = gs.fit(x,y)

    df = pd.DataFrame(result.cv_results_)
    df.to_csv('gridResults.csv')

    print("\nIt has score (on cross-validation):", result.best_score_)
    print("\nIt has parameters :", result.best_params_)

    return result.best_estimator_ #returning the best model

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################


#######################################################################
#                       ** EPOCH **
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
        # print("Currently at epoch:", epoch)
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

#######################################################################
#                       * IMPACT OF LAYERS GRAPH *
#######################################################################
def graph_layers():
    output_label = "median_house_value"
    data = pd.read_csv("housing.csv") 
    data = data.sample(frac=1).reset_index(drop=True)
    x= data.loc[:, data.columns != output_label]
    y = data.loc[:, [output_label]]
    split_idx = int(0.8 * len(x))
    x_train = x[:split_idx]
    y_train = y[:split_idx]
    x_test = x[split_idx:]
    y_test = y[split_idx:]

    epochs = []
    results = []
    for num_layers in range(1, 11, 2):
        results.append([])
        
        print("Layer:", num_layers + 1)

        for epoch in range(100, 1001, 100):
            print("-- Epoch:", epoch)
            if num_layers == 1: # i only want to do this once
                epochs.append(epoch)
            
            in_neurons = ([20] * num_layers)
            in_neurons.append(1)
            in_activations = (["relu"] * num_layers)
            in_activations.append("identity")

            reg = Regressor(x=x, nb_epoch=epoch, neurons=in_neurons, activations=in_activations)

            reg.fit(x_train, y_train)
            results[-1].append(reg.score(x_test,y_test))


    plt.figure(figsize=(8,6))
    for index in range(len(results)):
        plt.plot(epochs, results[index], label=(str(index + 1)+' layer'))

    plt.title("RMSE per epoch for different number of layers")
    plt.xlabel("Epochs")
    plt.ylabel("RMSE loss on test set")
    plt.legend()
    file_name = "graphs/LayerGraph"
    plt.savefig(file_name)
    plt.clf() # Clears the figure so the graphs don't overlap in the saved file


def graph_learning_rate():
    output_label = "median_house_value"
    data = pd.read_csv("housing.csv") 
    data = data.sample(frac=1).reset_index(drop=True)
    x= data.loc[:, data.columns != output_label]
    y = data.loc[:, [output_label]]
    split_idx = int(0.8 * len(x))
    x_train = x[:split_idx]
    y_train = y[:split_idx]
    x_test = x[split_idx:]
    y_test = y[split_idx:]

    epochs = []
    results = []
    rate = []
    for ten_lr in range(1, 6, 1):
        results.append([])
        learning_rate = ten_lr / 10
        rate.append(learning_rate)
        
        print("Learning rate:", learning_rate)

        for epoch in range(100, 1001, 100):
            print("-- Epoch:", epoch)
            if ten_lr == 1: # i only want to do this once
                epochs.append(epoch)

            reg = Regressor(x=x, nb_epoch=epoch, learning_rate=learning_rate)

            reg.fit(x_train, y_train)
            results[-1].append(reg.score(x_test,y_test))


    plt.figure(figsize=(8,6))
    for index in range(len(results)):
        plt.plot(epochs, results[index], label=(str(rate[index])+' learning rate'))

    plt.title("RMSE per epoch for different values of learning rate")
    plt.xlabel("Epochs")
    plt.ylabel("RMSE loss on test set")
    plt.legend()
    file_name = "graphs/LearningGraph"
    plt.savefig(file_name)
    plt.clf() # Clears the figure so the graphs don't overlap in the saved file



#######################################################################
#                       **  MAIN **
#######################################################################

def dummy_main():
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

    regressor = Regressor(x_train)
    regressor.fit(x_train, y_train)
    # save_regressor(regressor)

    # reg = load_regressor()

    # # Error
    error = regressor.score(x_test,y_test)
    print("\nRegressor error: {}\n".format(error))

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

    save_regressor(RegressorHyperParameterSearch(x_train, y_train))

    reg = load_regressor()

    # # Error
    error = reg.score(x_test,y_test)
    print("\nRegressor error: {}\n".format(error))


if __name__ == "__main__":
    dummy_main()
    #example_main()

    # epochs, dropout_eval_errors, no_dropout_eval_errors, dropout_test_errors, no_dropout_test_errors = overfitting_analysis()

    # print("no_dropout_eval_errors", no_dropout_eval_errors)
    # print("dropout_eval_errors",dropout_eval_errors)
    
    # print("no_dropout_test_errors", no_dropout_test_errors)
    # print("dropout_test_errors", dropout_test_errors)
    

    # graph_it(epochs, dropout_eval_errors, no_dropout_eval_errors, dropout_test_errors, no_dropout_test_errors)
