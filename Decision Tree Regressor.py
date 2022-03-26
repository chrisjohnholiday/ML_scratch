#!/usr/bin/env python
# coding: utf-8

# In[1]:


####################################################################
# Decision_Tree_Regressor.py                                       #
#                                                                  #
# By: Christopher Holiday                                          #
# Due: April 1st 2021                                              #
####################################################################

## Largely inspired by Riz Ozcelik 
## https://towardsdatascience.com/regression-trees-from-scratch-in-30-lines-of-python-8196e0ab29b2

####################################################################
#  Managing Dependencies                                           #
####################################################################

import numpy as np
import time
from random import shuffle
from random import seed
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import load_boston

seed(1001345760)

####################################################################
#  Loading Data                                                    #
####################################################################

boston = load_boston()
features_boston = boston['data']
classes_boston = boston['target']


# In[2]:


####################################################################
#  Creating classifier from scratch                                #
####################################################################


class holidayRegressionTree():
    

    def createTree(self, x_vars, y_var, tree_depth, min_split):
       ## This is a recursive function, stopping point is tree depth or not ##
       ## enough observations to split on ##
        
        ## Stop if tree has reached depth or if there aren't enough observations ##
        if tree_depth == 0 or len(y_var)  < min_split:
            return {"value": np.mean(y_var)}
        
        ## Initialize a best parameter search ##
        node_var = None
        node_threshold = None
        min_rss = np.inf
        under_index = None
        over_index = None
        ## Iterate over each features and threshold values to minimize RSS for both datasets##
        for i in range(len(x_vars[0])):
            ## Get thresholds for splitting node from variable levels##
            thresholds = np.unique(x_vars[:,i])
            for j in range(len(thresholds)):
                ## Define which outcomes are above and below threshold ##
                y_under =  y_var[x_vars[:,i] < thresholds[j]] 
                y_over = y_var[x_vars[:,i] >= thresholds[j]]
                ## Calculate RSS
                sum_rss = np.sum(np.square(y_under - np.mean(y_under))) + np.sum(np.square(y_over - np.mean(y_over)))
                    
                ## Store threshold and variable if it beats the minimum RSS
                if sum_rss < min_rss:
                    node_var = i 
                    node_threshold = thresholds[j]
                    ## This is the new RSS to beat ##
                    min_rss = sum_rss

        ## Set splitting node and splitting threshold ##
        tree = {"node_var": node_var, "node_threshold": node_threshold}
        ## Get correct indexes ##
        under_index = x_vars[:, node_var] < node_threshold
        over_index = x_vars[:, node_var] >= node_threshold
        
        ## Recursively call back function to split on a new node##
        tree["under"] = self.createTree(x_vars[under_index], y_var[under_index], tree_depth - 1, min_split)
        tree["over"] = self.createTree(x_vars[over_index], y_var[over_index], tree_depth - 1, min_split)
        
        self.tree = tree
        return tree
        
    
    ## Value prediction function ##
    def predict_value(self, x_to_predict, tree):
        ## Recursively search through tree for predicted value ##
        if "value" in tree:
            ## Get value if the node is terminal##
            return tree["value"]
        else:
            ## otherwise search through subtree, lower or higher ##
            if x_to_predict[tree["node_var"]] < tree["node_threshold"]:
                ## Recursive search below threshold ##
                return self.predict_value(x_to_predict, tree["under"])
            else:
                ## Recursive search at threshold ##
                return self.predict_value(x_to_predict, tree["over"])
        
        
    ## Create a fit object so we can use it like Sklearn##
    def fit(self, x_vars, y_var, max_depth = 4, min_split = 2):
        ## Make sure the depth is positive integer ##
        assert max_depth > 0
        ## Return the fitted tree object ##
        self.tree = self.createTree(x_vars, y_var, max_depth, min_split)
        return self
    
    ## Generate predictions with fit object
    def predict(self, x_array):
        ## Assert input shape/type correct ##
        assert isinstance(x_array, np.ndarray), "predicted input must be Numpy ndarray"
        assert len(x_array.shape) <= 2, "X variables must be a two dimensional Numpy ndarray or single observation"
        ## Generate empty predicted values 
        N_to_predict = x_array.shape[0]
        predictions = np.zeros(N_to_predict)
        
        ## Generate predictions for each x observation fitted##
        for predicting_i in range(N_to_predict):
            predictions[predicting_i] = self.predict_value(x_array[predicting_i], self.tree)
            
        return predictions        


# In[3]:


#########################################################
##        K-Fold Validate using MSE as the outcome     ##
#########################################################


## This is a K-fold validation procedure for decision trees that reports train and test MSE ##
def kFoldValidateMSE(folds, x_variables, y_variables, prediction_model, max_depth, min_split):
    
    ## Assert input types ##
    assert isinstance(folds, int), "Number of folds must be an integer"
    assert isinstance(x_variables, np.ndarray), "X variables must be Numpy ndarray"
    assert isinstance(y_variables, np.ndarray), "y variables must be Numpy ndarray"
    
    ## Assert input dimensions and lengths match ##
    assert len(x_variables.shape) <= 2, "X variables must be a two dimensional Numpy ndarray"
    assert len(y_variables.shape) == 1, "Y variables must a one dimensional Numpy ndarray"
    assert x_variables.shape[0] == y_variables.shape[0], "Length of first dimension for X and Y variables do not match"
     
    ## Assert that number of folds is possible ##
    assert folds <= len(y_variables), "Folds must be less than the number of observations"
    
    ## Generate random sequence of variables ##
    N_obs = len(y_variables)
    sequence = [i for i in range(N_obs)]
    rand_sequence = shuffle(sequence)
    sequenced_y = y_variables[rand_sequence][0]
    sequenced_x = x_variables[rand_sequence][0]
    
    ## Create fold boundaries and accuracy metrics for train and test##
    fold_size = int(N_obs / folds)
    accuracy = np.zeros([folds, 2])
    
    ## Run Model on the first k-1 folds 
    for k_fold in range(folds - 1):
        
        ## Set indexes for training and testing in this fold ## 
        idx_start = k_fold * fold_size
        idx_end = (k_fold + 1) * fold_size
        ## Create index test and train array ##
        idx_test = np.array(range(idx_start, idx_end))
        idx_train = np.delete(np.array(range(N_obs)), idx_test)
        
        ## Run model and make predictions on train/test set ##
        fitted_model = prediction_model.fit(sequenced_x[idx_train], sequenced_y[idx_train], max_depth, min_split)
        ## Get predictions for train and test dataset ##
        predictions_train = fitted_model.predict(sequenced_x[idx_train])
        predictions_test = fitted_model.predict(sequenced_x[idx_test])
        
        ## Score each of the predictions in terms of MSE##
        accuracy[k_fold, 0] = np.mean(np.square(sequenced_y[idx_train] - predictions_train))
        accuracy[k_fold, 1] = np.mean(np.square(sequenced_y[idx_test] - predictions_test))
    
    ## Last fold we repeat the same procedure ##
    idx_test_last = np.array(range((folds - 1) * fold_size , N_obs))
    idx_train_last = np.array(range((folds - 1) * fold_size))

    ## Run model and make predictions on train/test set last time ##
    fitted_model = prediction_model.fit(sequenced_x[idx_train_last], sequenced_y[idx_train_last], max_depth, min_split)
    predictions_train = fitted_model.predict(sequenced_x[idx_train_last])
    predictions_test = fitted_model.predict(sequenced_x[idx_test_last])

    ## Score each of the predictions in terms of MSE ##
    accuracy[folds - 1, 0] = np.mean(np.square(sequenced_y[idx_train_last] - predictions_train))
    accuracy[folds - 1, 1] = np.mean(np.square(sequenced_y[idx_test_last] - predictions_test))
    
    ## Return accuracy as 2 x fold array, 1st column is train error second is test ##
    return accuracy


# In[4]:


## Conduct a grid search for best tree pre-pruning level ##
def gridSearchTree(folds, x_vars, y_var, clf, max_max_depth = 10, max_min_split = 10):
    ## Baseline model ##
    accuracy = np.inf
    best_max_depth = None
    best_min_split = None
    
    ## Iterate to find the best parameter values ##
    for max_depth_i in range(2, max_max_depth):
        for min_split_j in range(2, max_min_split):
            accuracy_iteration = kFoldValidateMSE(folds, x_vars, y_var, clf, max_depth_i, min_split_j)
            ## Store hyperparameters if it's more accurate ##
            if np.mean(accuracy_iteration[:,1]) < accuracy:
                accuracy = np.mean(accuracy_iteration[:,1])
                best_max_depth = max_depth_i
                best_min_split = min_split_j
    
    ## Return best tree model and report best hyperparameter results ##
    print("Lowest MSE:", accuracy, "Max depth:", best_max_depth, "Min split:", best_min_split)
    return clf.fit(x_vars, y_var, best_max_depth, best_min_split)


# In[5]:


#########################################################
##        Compare SKlearn with my best tree            ##
#########################################################

## Fit my regression tree first I used a grid search to find optimal values ##
clf = holidayRegressionTree()
best_tree = gridSearchTree(4, features_boston, classes_boston, clf, 10, 3)

## Time the regression tree at the best optiaml points##
holiday_tree_start = time.time()
clf = holidayRegressionTree()
holiday_tree = clf.fit(features_boston, classes_boston, 5, 2)
holiday_tree_end = time.time()


# In[6]:


## Fit sklearn decision tree next ##
## I used the best parameters that I found when gridsearching on my regression tree to make ##
## Them comparable ##
 
sklearn_tree_start = time.time()
clf = DecisionTreeRegressor(max_depth = 5)
sklearn_best_tree = clf.fit(features_boston, classes_boston)
sklearn_tree_end = time.time()

## Compare the two times ##
print("My Regression tree took ", np.round(holiday_tree_end - holiday_tree_start, 6), " time to run.")
print("Sklearn Regression tree took ", np.round(sklearn_tree_end - sklearn_tree_start, 6), " time to run.")

## Compare the average MSE of each model on the full dataset ##
my_tree_mse = np.mean(np.square(classes_boston - best_tree.predict(features_boston)))
sklearn_tree_mse = np.mean(np.square(classes_boston - sklearn_best_tree.predict(features_boston)))

## Report results ##
print("My tree achieved an MSE of", np.round(my_tree_mse, 6), "Sklearn achieved an MSE of", np.round(sklearn_tree_mse, 6))

## At optimal depth my desicion tree produces identical performance in MSE ##
## Mine takes significantly longer to converge, guess I have to learn C! ##
#########################################################

