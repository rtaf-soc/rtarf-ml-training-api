import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.compose import make_column_transformer

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

from sharelib import *
########### mflow ############
import mlflow
import mlflow.sklearn
########### mflow ############

import logging

from pathlib import Path
import json
import glob

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

if __name__ == "__main__":

    print("1111111111")
    
    
    df = pd.DataFrame()
    path_to_json = 'rawdata' 
    print(path_to_json)
    json_pattern = os.path.join(path_to_json,'*.txt')
    print(json_pattern)    
    file_list = glob.glob(json_pattern)
    print(file_list)
    
    for file in file_list:
        data = pd.read_json(file, lines=True)
        df = pd.concat([df,data], ignore_index = True)
    
    