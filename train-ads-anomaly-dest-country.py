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
    
    xcountDetect = 0

    for file in file_list:
        xcountDetect = xcountDetect + 1
        print("xcountDetect: ", xcountDetect)
        if (xcountDetect == 10):
            break
        data = pd.read_json(file, lines=True)
        df = pd.concat([df,data], ignore_index = True)

    
    
    np.set_printoptions(threshold=sys.maxsize)

    df_categories = df[df["ads_countDetectry_dst"].str.startswith(('1', '2', '3', '4', '5', '6', '7', '8', '9', '0')) == False] 
    df_categories = pd.concat([df_categories["ads_countDetectry_dst"]], axis=1, sort=False,)
    
    countDetectryStr = listOfCountryDst()
    print("-------------- Show Country Not in list --------------")
    print(df_categories[~df_categories['ads_countDetectry_dst'].isin(countDetectryStr)])
    print("-------------- Show Country Not in list --------------")

    df_categories = df_categories.mask(~df_categories.isin(countDetectryStr),'OTHER')
    X_transform = createXTransformOrdinalDst()
    X = X_transform.transform(df_categories)
    
    # Call and fit the Local Outlier Factor detector
    lof_detector = LocalOutlierFactor(n_neighbors=10, contamination=0.01,novelty=True).fit(X.values)
    lof_detect = lof_detector.predict(X)

    recordDetect,countDetect = np.unique(lof_detect, return_countDetects=True)
    print("--------------Count Anomaly VS Normal-------------")
    print("Anomaly = " , countDetect[0] , "record with " , (countDetect[0])*100/(countDetect[0]+countDetect[1]) ," %")
    print("Normal  = " , countDetect[1] , "record with " , (countDetect[1])*100/(countDetect[0]+countDetect[1]) ," %")
    print("--------------Count Anomaly VS Normal-------------")
    
    print("-------------- List Destination Country with Prediction -------------")
    # print(type(df_categories.value_countDetects()))
    # print(type(df_categories))
    for index, value in df_categories.value_countDetects().items():
        tempdf = pd.DataFrame([
            [index[0]]
        ], columns=['ads_countDetectry_dst'])
        encode = X_transform.transform(tempdf)
        predictData = lof_detector.predict(encode)
        print(index[0]  , " | code =" , encode.values[0][0] , " | countDetect ="  , value , " | result =" , dataPredictionToString(predictData[0]))

    print("-------------- List Destination Country with Prediction -------------")

    plt.figure(figsize=(20,20))
    plt.scatter(X.to_numpy()[:, 0], X.to_numpy()[:, 0], c=lof_detect, cmap="flag", alpha=0.5)
    plt.title("LocalOutlierFactor")
    plt.show()

    tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
    # export MLFLOW_TRACKING_USERNAME=user 
    # export MLFLOW_TRACKING_PASSWORD=pwd

    experiment = mlflow.get_experiment_by_name('ads-anomaly-dest-countDetectry')
    experiment_id = experiment.experiment_id

    with mlflow.start_run(experiment_id=experiment_id):
        mlflow.set_tracking_uri(tracking_uri)
        
        print("Artifact Location: {}".format(experiment.artifact_location))
        print("artifact uri : " + mlflow.get_artifact_uri())

        mlflow.doctor()

        mlflow.log_metric("Anomaly", str((countDetectDetect[0])*100/(countDetectDetect[0]+countDetectDetect[1])))
        mlflow.log_metric("Normal", str((countDetectDetect[1])*100/(countDetectDetect[0]+countDetectDetect[1])))
        mlflow.sklearn.log_model(lof_detector, "model", registered_model_name="soc-ml")
        print("Model saved in run %s" % mlflow.active_run().info.run_uuid)


    # Plot the conparison between actual and predicted y
    df_categories.value_countDetects()[: :].plot(kind="bar", figsize=(20,20))
    plt.show()