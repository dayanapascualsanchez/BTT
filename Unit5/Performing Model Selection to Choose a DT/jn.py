import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import types
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import sklearn
from sklearn.metrics import accuracy_score

#########
#
# First Test
#
def testXY(y, X, df):
    
    if not isinstance(df, pd.DataFrame):
        return (False,'Incorrect!\nSomething is not quite right with DataFrame df. Go back and make sure df was properly created.')

    y_result = df['Churn'] 
    x_result = df.drop(columns = 'Churn', axis=1)

    if not isinstance(y, pd.Series):
        return(False, 'Incorrect!\y is incorrect. It should be of type Series.')
    
    if not isinstance(X, pd.DataFrame):
        return (False,'Incorrect!\X is incorrect. It should be of type DataFrame.')
     
    
    errorString = []
    if not y.equals(y_result):
        errorString.append('Incorrect!\y is incorrect. It does not have the correct values.')

    if not X.equals(x_result):
        errorString.append('Incorrect!\X is incorrect. It does not have the correct values.')

    if errorString:
        return (False,"\n".join(errorString))
        
    return (True, "Correct!")

#########
#
# Second Test
#
def testSplit(X_train_student, X_test_student, y_train_student, y_test_student, df):

    # check types
    if not isinstance(df, pd.DataFrame):
        return (False,'Incorrect!\nSomething is not quite right with DataFrame df. Go back and make sure df was properly created.')

    errorString = []
    e1 = 'Make sure you are calling train_test_split() with the proper arguments in the right order.'

    # check values
    y = df['Churn'] 
    X = df.drop(columns = 'Churn', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=1234)    
    #X_train check type, shape, value
    if not isinstance(X_train_student, pd.DataFrame):
        errorString.append('Incorrect!\nX_train is incorrect. It should be of type DataFrame.')   
    if not X_train_student.shape==(45942, 84):
        errorString.append("Incorrect!\nX_train is incorrect. It does not have the proper shape.")
    if not X_train_student.equals(X_train):
        errorString.append('Incorrect!\nX_train is incorrect. It does not have the correct values.')

    #X_test check type, shape, value
    if not isinstance(X_test_student, pd.DataFrame):
        errorString.append('Incorrect!\nX_test is incorrect. It should be of type DataFrame.')
    if not X_test_student.shape==(5105, 84):
        errorString.append("Incorrect!\nX_test is incorrect. It does not have the proper shape.")
    if not X_test_student.equals(X_test):
        errorString.append('Incorrect!\nX_test is incorrect. It does not have the correct values.')

    #y_train check type, shape, value
    if not isinstance(y_train_student, pd.Series):
        errorString.append('Incorrect!\ny_train is incorrect. It should be of type Series.')
    if not y_train_student.shape==(45942,):
        errorString.append("Incorrect!\ny_train is incorrect. It does not have the proper shape.")
    if not y_train_student.equals(y_train):
        errorString.append('Incorrect!\ny_train is incorrect. It does not have the correct values.')

    #y_test check type, shape, value
    if not isinstance(y_test_student, pd.Series):
        errorString.append('Incorrect!\ny_test is incorrect. It should be of type Series.')
    if not y_test_student.shape==(5105,):
        errorString.append("Incorrect!\ny_test is incorrect. It does not have the proper shape.")
    if not y_test_student.equals(y_test):
        errorString.append('Incorrect!\ny_test is incorrect. It does not have the correct values.')
    
    if errorString:
        errorString.append('Hint: Make sure you are calling train_test_split() with the proper arguments in the right order.')
        return (False,"\n".join(errorString))
        
    
    return (True, "Correct!")

#########
#
# Third Test
#
def testModelSelection(df, accuracy_scores, first=True):

    # check types
    if not isinstance(df, pd.DataFrame):
        return (False,'Incorrect!\nSomething is not quite right with DataFrame df. Go back and make sure df was properly created.')
    
    # compute the result
    hyperparams = [2**n for n in range(2,5)]
    y = df['Churn'] 
    X = df.drop(columns = 'Churn', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=1234)

    accuracy_scores_results = []

    if first:
        print('Running Test.....')    
    for md in hyperparams:
        model = DecisionTreeClassifier(max_depth = md, min_samples_leaf = 1)
        acc_score = cross_val_score(model, X_train, y_train, cv=5)
        acc_mean = acc_score.mean()
        accuracy_scores_results.append(acc_mean)
    
    if first:
        print('Test Complete. See results below:\n\n')

    if not np.allclose(np.array(accuracy_scores_results), np.array(accuracy_scores),atol=1e-01):
        return(False, 'Incorrect!\naccuracy_scores is incorrect.')
        
    return (True, "Correct!")

#########
#
# Fourth Test
#
def testDTModel(df, model, class_label_predictions, acc_score):

    if not isinstance(df, pd.DataFrame):
        return (False,'Incorrect!\nSomething is not quite right with DataFrame df. Go back and make sure df was properly created.')
    
    if not isinstance(model, sklearn.tree._classes.DecisionTreeClassifier):
        return (False,'Incorrect!\nYour model object is not a of type DecisionTreeClassifier.\nCheck how you are calling DecisionTreeClassifier().')

    errorString = []
    if not model.min_samples_leaf == 1:
        errorString.append('Incorrect!\nThe hyperparameter min_samples_leaf should be 1.\nCheck how you are calling DecisionTreeClassifier() and specifying min_samples_leaf.')
    if not model.max_depth == 4:
        errorString.append('Incorrect!\nThe hyperparameter max_depth does not contain the correct value.\nUse the value of max_depth that resulted in the best accuracy score.')

    if errorString:
        return(False, "\n".join(errorString))

    y = df['Churn'] 
    X = df.drop(columns = 'Churn', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=1234)
    model = DecisionTreeClassifier(max_depth = 4, min_samples_leaf = 1)
    model.fit(X_train, y_train)
    class_lp_result = model.predict(X_test)
    acc_score_result = accuracy_score(y_test, class_label_predictions)

    if not np.array_equal(class_lp_result, class_label_predictions):
        return (False,'Icorrect!\nclass_label_predictions not contain the correct values.\nCheck how you are calling DecisionTreeClassifier(), .fit() and .predict().')

    if not np.isclose(acc_score, acc_score_result, atol=0.01):
        return (False,'Incorrect!\nacc_score is not correct.')
        
    return (True, "Correct!")

#########
#
# Fifth Test
#
def testGridSearch(model, grid, grid_search):

    hyperparams_depth = [2**n for n in range(2,5)]
    hyperparams_leaf = [25*2**n for n in range(0,3)]
    param_grid_result={'max_depth':hyperparams_depth, 'min_samples_leaf':hyperparams_leaf}
    cv = 5

    if not isinstance(model, sklearn.tree._classes.DecisionTreeClassifier):
        return (False,'Incorrect!\nYour model object is not a of type DecisionTreeClassifier.\nCheck how you are calling DecisionTreeClassifier().')
    if not isinstance(grid, sklearn.model_selection._search.GridSearchCV):
        return (False,'Incorrect!\nVariable grid is not a of type GridSearchCV.\nCheck how you are calling GridSearchCV() and which arguments you are using.')
    if not isinstance(grid.estimator, sklearn.tree._classes.DecisionTreeClassifier):
        return (False,'Incorrect!\nThe arguments to GridSearchCV() are not correct.\nMake sure to pass your DecisionTreeClassifier model object as an argument.')
    if not grid.param_grid == param_grid_result:
        return (False,'Incorrect!\nThe arguments to GridSearchCV() are not correct.\nMake sure to pass the param_grid dictionary as an argument.')
    if not grid.cv == 5:
        return (False,'Incorrect!\nThe arguments to GridSearchCV() are not correct.\nMake sure to use the parameter cv=5.')

    if not isinstance(grid_search, sklearn.model_selection._search.GridSearchCV):
        return (False,'Incorrect!\nVariable grid_search is not correct. \nCheck how you are calling the grid.fit() on the training data.')

    return (True, "Correct!")


