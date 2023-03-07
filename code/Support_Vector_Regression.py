
#1 Importing the libraries
import json
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy import savetxt
import argparse
import os
import glob
output_file = "predict_output.csv"

def parse_args() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, default="/opt/ml/processing")
    args, _ = parser.parse_known_args()
    return args

if __name__ == "__main__":
    print("Starting job")
    args = parse_args()
    base_dir = args.base_dir
    input_dir = os.path.join(base_dir, "data")
    
    input_file_list = glob.glob(f"{input_dir}/*.csv")
    #2 Concat input files with select columns
    df = []
    for file in input_file_list:
        df_tmp = pd.read_csv(file)
        df.append(df_tmp)
    dataset = pd.concat(df, ignore_index=True)
        
    print("Data loaded in to a dataframe")

        
    X = dataset.iloc[:,1:2].values.astype(float)
    y = dataset.iloc[:,2:3].values.astype(float)

    #3 Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc_X = StandardScaler()
    sc_y = StandardScaler()
    X = sc_X.fit_transform(X)
    y = sc_y.fit_transform(y)

    #4 Fitting the Support Vector Regression Model to the dataset
    # Create your support vector regressor here
    from sklearn.svm import SVR
    # most important SVR parameter is Kernel type. It can be linear,polynomial or gaussian.
    #SVR. We have a non-linear condition so we can select polynomial or gaussian but here
    #we select RBF(a gaussian type) kernel. 
    regressor = SVR(kernel='rbf')
    regressor.fit(X,y)

    #5 Predicting a new result
    y_pred = sc_y.inverse_transform((regressor.predict(sc_X.transform(np.array([[6.5]])))))
    
    X_grid = np.arange(min(X), max(X), 0.1)
    X_grid = X_grid.reshape((len(X_grid), 1))

    y_out = regressor.predict(X_grid)
    savetxt(f"{base_dir}/output/{output_file}", y_out, delimiter=',')
    print("finish processing job")
