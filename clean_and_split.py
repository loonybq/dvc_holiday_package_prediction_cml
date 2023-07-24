import sys
import os

import pandas as pd
from sklearn.model_selection import train_test_split

def clean_and_split(input_data_folder):
	package_sale_data = pd.read_csv(f'{input_data_folder}/holiday_package_data.csv')

	package_sale_data = package_sale_data.drop(columns = ['CustomerID'])
	package_sale_data.replace('Fe Male','Female', inplace = True)

	X = package_sale_data.drop(columns = ['ProdTaken'])
	y = package_sale_data['ProdTaken']

	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123)

	# Create folder to save file
	data_path = 'split_data'
	os.makedirs(data_path, exist_ok = True)

	X_train.to_csv('split_data/train_features.csv', index=False)
	X_test.to_csv('split_data/test_features.csv', index=False)
	y_train.to_csv('split_data/train_labels.csv', index=False)
	y_test.to_csv('split_data/test_labels.csv', index=False)


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Please provide the name of the input data folder")
    else:
        input_data_folder = sys.argv[1]
        
        clean_and_split(input_data_folder)

