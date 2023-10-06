import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor

########### mflow ############
import mlflow
import mlflow.sklearn
########### mflow ############

from sklearn.preprocessing import OrdinalEncoder

from sharelib import *
# import logging
from pathlib import Path
import json
import glob
import sys

# logging.basicConfig(level=logging.WARN)
# logger = logging.getLogger(__name__)


if __name__ == "__main__":
    df = pd.DataFrame()
    path_to_json = 'rawdata' 
    json_pattern = os.path.join(path_to_json,'*.txt')
    file_list = glob.glob(json_pattern)
    
    xcount = 0

    for file in file_list:
        xcount = xcount + 1
        print("xcount: ", xcount)
        data = pd.read_json(file, lines=True)
        df = pd.concat([df,data], ignore_index = True)

    
        np.set_printoptions(threshold=sys.maxsize)
    
    
    df_categories = df[df["ads_country_dst"].str.startswith(('1', '2', '3', '4', '5', '6', '7', '8', '9', '0')) == False] 
    df_categories = pd.concat([df_categories["ads_country_dst"]], axis=1, sort=False,)
    
    countryStr = listOfCountryDst()
    df_categories = df_categories.mask(~df_categories.isin(countryStr),'OTHER')
   
    X_transform = createXTransformOrdinalDst()
    print(X_transform.categories_)
    X = X_transform.transform(df_categories)
    
    print(X)
    
