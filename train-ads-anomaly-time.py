import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor

########### mflow ############
import mlflow
import mlflow.sklearn
########### mflow ############

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder

from sharelib import *
import logging
from pathlib import Path
import json
import glob
import sys
import pickle

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

jenkinsURL = getArgs(1,"")
mlflowMinioFolder = getArgs(2,"")
mlflowTrainingFileLimit = int(getArgs(3,10))

# print(mlflowTrainingFileLimit)

if __name__ == "__main__":
    df = pd.DataFrame()
    path_to_json = 'rawdata6' 
    json_pattern = os.path.join(path_to_json,'*.txt')
    file_list = glob.glob(json_pattern)
    
    xcount = 0

    for file in file_list:
        xcount = xcount + 1
        print("xcount: ", xcount)
        data = pd.read_json(file, lines=True)
        df = pd.concat([df,data], ignore_index = True)
        if (xcount == mlflowTrainingFileLimit):
            break
    

        np.set_printoptions(threshold=sys.maxsize)
    
    df_categories = pd.concat([df["ads_ts_hh"]], axis=1, sort=False,)
    print("-------------- Count Record --------------")
    print(df_categories.shape[0])
    print("-------------- Count Record --------------")
    print("-------------- Count HH --------------")
    print(df_categories.value_counts().to_string())
    print("-------------- Count HH --------------")

    X = df_categories

    # Call and fit the Local Outlier Factor detector
    lof_detector = LocalOutlierFactor(n_neighbors=100, contamination=0.1,novelty=True).fit(X.values)

    print("-------------- Model Size (MB) --------------")
    print("{:.2f}".format(sys.getsizeof(pickle.dumps(lof_detector))/(1024*1024)))
    print("-------------- Model Size (MB) --------------")

    lof_detect = lof_detector.predict(X)

    recordDetect,countDetect = np.unique(lof_detect, return_counts=True)
    print("--------------Count Anomaly VS Normal-------------")
    print(recordDetect)
    print(countDetect)

    if (len(countDetect) == 1):
        row_to_be_added = countDetect
        countDetect = np.append(np.array([0]),row_to_be_added,axis=0)

    print("Anomaly = " , countDetect[0] , "record with " , (countDetect[0])*100/(countDetect[0]+countDetect[1]) ," %")
    print("Normal  = " , countDetect[1] , "record with " , (countDetect[1])*100/(countDetect[0]+countDetect[1]) ," %")
    print("--------------Count Anomaly VS Normal-------------")
    # print(lof_detect)

    np.set_printoptions(formatter={'float_kind':'{:f}'.format})
    print(np.unique(lof_detector.negative_outlier_factor_, return_counts=True))
    print(lof_detector.negative_outlier_factor_)

    print("-------------- List HH with Prediction -------------")
    for index, value in df_categories.value_counts().items():
        tempdf = pd.DataFrame([
            [index[0]]
        ], columns=['ads_ts_hh'])
        predictData = lof_detector.predict(tempdf)
        print(index[0]  , " | count ="  , value , " | result =" , dataPredictionToString(predictData[0]))

    print("-------------- List HH with Prediction -------------")

    plt.figure(figsize=(20,20))
    plt.scatter(X.to_numpy()[:, 0], X.to_numpy()[:, 0], c=lof_detect, cmap="flag", alpha=0.5)
    plt.title("LocalOutlierFactor")
    plt.show()

    tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
    # export MLFLOW_TRACKING_USERNAME=user 
    # export MLFLOW_TRACKING_PASSWORD=pwd

    experiment = mlflow.set_experiment(experiment_name='ads-anomaly-time')
    experiment_id = experiment.experiment_id

    run_description = f"""
### Note
**All information** * about Training * ~~ML here~~
Jenkins URL: [{jenkinsURL}]({jenkinsURL})
    """

    with mlflow.start_run(experiment_id=experiment_id,description=run_description):
        mlflow.set_tracking_uri(tracking_uri)
        
        print("Artifact Location: {}".format(experiment.artifact_location))
        print("artifact uri : " + mlflow.get_artifact_uri())

        mlflow.doctor()
        mlflow.log_param("MlflowMinioFolder", mlflowMinioFolder)
        mlflow.log_param("SampleFile", xcount)
        mlflow.log_param("SampleRows", X.shape[0])
        # mlflow.set_tag("JenkinsURL",jenkinsURL)

        mlflow.log_metric("Anomaly", str((countDetect[0])*100/(countDetect[0]+countDetect[1])))
        mlflow.log_metric("Normal", str((countDetect[1])*100/(countDetect[0]+countDetect[1])))
        mlflow.sklearn.log_model(lof_detector, "model", registered_model_name="ads-anomaly-by-time")
        print("Model saved in run %s" % mlflow.active_run().info.run_uuid)