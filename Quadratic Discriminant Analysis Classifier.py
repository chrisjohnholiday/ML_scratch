####################################################################
# Quadratic_Discriminant_Analysis_Classifier.py                    #
#                                                                  #
# By: Christopher Holiday 1001345760 holidayc                      #
# Due: April 1st 2021                                              #
####################################################################

## Heavily inspired from Xavier Bourret Sicotte:
## https://xavierbourretsicotte.github.io/LDA_QDA.html

####################################################################
#  Managing Dependencies                                           #
####################################################################

import numpy as np
import time
from random import shuffle
from random import seed
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.datasets import load_breast_cancer

seed(1001345760)

####################################################################
#  Loading Data                                                    #
####################################################################

breast_cancer = load_breast_cancer()
features_breast_cancer = breast_cancer['data']
classes_breast_cancer = breast_cancer['target']



####################################################################
#  Creating classifier from scratch                                #
####################################################################

class holiday_QDA():
        
    ## Defining basic functions used in object ##
    def fit_gaussian(self, x_vars, y_var):

        ## Assert input types ##
        assert isinstance(x_vars, np.ndarray), "X variables must be Numpy ndarray"
        assert isinstance(y_var, np.ndarray), "y variables must be Numpy ndarray"

        ## Assert input dimensions and lengths match ##
        assert len(x_vars.shape) <= 2, "X variables must be a two dimensional Numpy ndarray"
        assert len(y_var.shape) == 1, "Y variables must a one dimensional Numpy ndarray"
        assert x_vars.shape[0] == y_var.shape[0], "Length of first dimension for X and Y variables do not match"


        ## get number of classes and features##
        N_classes = len(np.unique(y_var))
        N_features = x_vars.shape[1]

        ## Empty numpy arrays Averages, covariance matrix and probabilities for each label ##
        averages = np.zeros((N_classes, N_features))
        covariance = np.zeros((N_classes, N_features, N_features))
        probability = np.zeros(N_classes)

        ## Get averages, covariance matrix and probabilities for each label
        for k in range(N_classes):
            k_idx = (y_var == k)
            averages[k] = np.mean(x_vars[k_idx], axis = 0)
            covariance[k] = np.cov(x_vars[k_idx], rowvar = 0) ## May need a bias parameter?
            probability[k] = len(k_idx[k_idx == True])/len(y_var)

        return averages, covariance, probability

    ## Quadratic scoring function, that returns a score for each class(label) for every observation ##
    def quadratic_score_function(self, x_to_score, averages, covariance, probability):


        ## get number of classes and features##
        N_classes = len(probability)
        N_features = len(x_to_score)

        ## Use guassian to return scoring parameters for each
        scores = np.zeros(N_classes)

        for k in range(N_classes):
            covariance_inverse = np.linalg.inv(covariance[k])
            scores[k] = np.log(probability[k]) - 0.5 * np.log(np.linalg.det(covariance[k])) - 0.5 * np.matmul(np.matmul(np.transpose(x_to_score - averages[k]),covariance_inverse), x_to_score - averages[k])        
        return scores

    ## Function to get predicted label ##
    def predicted_label(self, x_to_score, averages, covariance, probability):

        ## Get scores for the label ##
        scores = self.quadratic_score_function(x_to_score, averages, covariance, probability)

        ## Return label index ##
        return np.argmax(scores)
    
    ## Fit model function
    def fit(self, x_vars, y_var):
        self.averages, self.covariance, self.probability = self.fit_gaussian(features_breast_cancer, classes_breast_cancer)
        return self 

    ## Generate predicted values for an array of X's ##
    def predict(self, x_array):
        
        ## Assert input shape/type correct ##
        assert isinstance(x_array, np.ndarray), "predicted input must be Numpy ndarray"
        assert len(x_array.shape) <= 2, "X variables must be a two dimensional Numpy ndarray or single observation"
        assert x_array.shape[1] == self.averages.shape[1], "X variables must be a two dimensional Numpy ndarray or single observation"
        
        ## Generate empty predicted values ##
        N_to_predict = x_array.shape[0]
        predictions = np.zeros(N_to_predict)
        
        ## Generate predictions for each ##
        for predicting_i in range(N_to_predict):
            predictions[predicting_i] = self.predicted_label(x_array[predicting_i], self.averages, self.covariance, self.probability)
        
        return predictions



####################################################################
#  K-Fold Cross Validation                                         #
####################################################################

def k_fold_validate(folds, x_variables, y_variables, prediction_model):
    
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
    
    ## Generate random sequence ##
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
        idx_test = np.array(range(idx_start, idx_end))
        idx_train = np.delete(np.array(range(N_obs)), idx_test)
        
        ## Run model and make predictions on train/test set ##
        fitted_model = prediction_model.fit(sequenced_x[idx_train], sequenced_y[idx_train])
        predictions_train = fitted_model.predict(sequenced_x[idx_train])
        predictions_test = fitted_model.predict(sequenced_x[idx_test])
        
        ## Score each of the predictions in terms of accuracy ratio ##
        accuracy[k_fold, 0] = (np.sum(sequenced_y[idx_train] == predictions_train)) / len(idx_train)
        accuracy[k_fold, 1] = (np.sum(sequenced_y[idx_test] == predictions_test)) / len(idx_test)

    ## Last fold ##
    idx_test_last = np.array(range((folds - 1) * fold_size , N_obs))
    idx_train_last = np.array(range((folds - 1) * fold_size))

    ## LAST: Run model and make predictions on train/test set ##
    fitted_model = prediction_model.fit(sequenced_x[idx_train_last], sequenced_y[idx_train_last])
    predictions_train = fitted_model.predict(sequenced_x[idx_train_last])
    predictions_test = fitted_model.predict(sequenced_x[idx_test_last])

    ## Score each of the predictions in terms of accuracy ratio ##
    accuracy[folds - 1, 0] = (np.sum(sequenced_y[idx_train_last] == predictions_train)) / len(idx_train_last)
    accuracy[folds - 1, 1] = (np.sum(sequenced_y[idx_test_last] == predictions_test)) / len(idx_test_last)
    
    return accuracy



####################################################################
#  Measuring peformance and accuracy between two QDA models one    #
#  by me and the other by SKlearn                                  #
####################################################################

for i in range(10):
    ## My model ##
    holiday_QDA_start = time.time()
    clf = holiday_QDA()
    accuracy_holiday_QDA = k_fold_validate(5, features_breast_cancer, classes_breast_cancer, clf)
    holiday_QDA_end = time.time()

    ## SKlearn Model ##
    sklearn_QDA_start = time.time()
    clf = QuadraticDiscriminantAnalysis()    
    accuracy_SKlearn_QDA = k_fold_validate(5, features_breast_cancer, classes_breast_cancer, clf);
    sklearn_QDA_end = time.time()

    ## Report the results of time ##
    print("My QDA took ", np.round(holiday_QDA_end - holiday_QDA_start, 6), " time to run.")
    print("Sklearn QDA took ", np.round(sklearn_QDA_end - sklearn_QDA_start, 6), " time to run.")

## Report training and test accuracies for both algorithims ##
print("My QDA function performance:")
for fold_iteration in range(len(accuracy_holiday_QDA)):
    print("On fold:", fold_iteration + 1, "Training accuracy was", np.round(accuracy_holiday_QDA[fold_iteration, 0], 6), "and testing accuracy was", np.round(accuracy_holiday_QDA[fold_iteration, 0], 6))

print("SKlearn function performance:")
for fold_iteration in range(len(accuracy_SKlearn_QDA)):
    print("On fold:", fold_iteration + 1, "Training accuracy was", np.round(accuracy_SKlearn_QDA[fold_iteration, 0], 6), "and testing accuracy was", np.round(accuracy_SKlearn_QDA[fold_iteration, 0], 6))

## My QDA generates similar levels of accuracy as SKLearn but takes significantly longer ##
####################################################################
