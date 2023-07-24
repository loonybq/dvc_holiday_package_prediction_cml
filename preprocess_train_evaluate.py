import sys
import os
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import ConfusionMatrixDisplay


def preprocess_train_evaluate(input_data_folder):

    # Read in data
    X_train = pd.read_csv(f'{input_data_folder}/train_features.csv')
    y_train = pd.read_csv(f'{input_data_folder}/train_labels.csv')
    X_test = pd.read_csv(f'{input_data_folder}/test_features.csv')
    y_test = pd.read_csv(f'{input_data_folder}/test_labels.csv')


    # Processing
    numeric_features = [
        'Age', 
        'DurationOfPitch', 
        'MonthlyIncome'
    ]
    numeric_transformer = Pipeline(
        steps = [
            ('imputer', SimpleImputer(strategy = 'median')), 
            ('scaler', StandardScaler())
        ]
    )
    categorical_features = [
        'TypeofContact', 
        'Occupation', 
        'Gender', 
        'ProductPitched', 
        'MaritalStatus', 
        'Designation'
    ]

    categorical_transformer = Pipeline(
        steps = [
            ('imputer', SimpleImputer(strategy = 'most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown = 'ignore'))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers = [
            ('num_tr', numeric_transformer, numeric_features),
            ('cat_tr', categorical_transformer, categorical_features)
        ], 
        remainder = SimpleImputer(strategy = 'most_frequent')
    )

    # Fit a model
    n_estimators =  20
    min_samples_split =  2
    min_samples_leaf  =  2

    rf_model = RandomForestClassifier(
        n_estimators = n_estimators, 
        min_samples_split = min_samples_split, 
        min_samples_leaf = min_samples_leaf, 
        class_weight = None
    )
    
    pipe_rf = Pipeline(
        steps = [('preprocessor', preprocessor), ('classifier', rf_model)]
    )
    pipe_rf.fit(X_train, y_train.values.ravel())

    y_pred =  pipe_rf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print('Accuracy: %.3f' % acc)
    print('Precision: %.3f' % prec)
    print('Recall: %.3f' % rec )
    print('F1 Score: %.3f' % f1)

    with open('metrics.json', 'w') as outfile:
        json.dump({'accuracy': acc, 'precision': prec, 'recall': rec, 'f1_score': f1}, outfile)

    # Plot it
    disp = ConfusionMatrixDisplay.from_estimator(
        pipe_rf, X_test, y_test, normalize = 'true', cmap = plt.cm.Blues
    )

    pd.DataFrame({
        'predicted': y_pred.squeeze(),
        'actual': y_test.squeeze()
    }).to_csv('predicted_vs_actual.csv', index=False)
    
    plt.savefig('plot.png')


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Please provide the name of the input data folder")
    else:
        input_data_folder = sys.argv[1]
        
        preprocess_train_evaluate(input_data_folder)








