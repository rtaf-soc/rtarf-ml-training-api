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
import pickle

# logging.basicConfig(level=logging.WARN)
# logger = logging.getLogger(__name__)

jenkinsURL = getArgs(1,"")
mlflowMinioFolder = getArgs(2,"")
mlflowTrainingFileLimit = int(getArgs(3,10))

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
        if (xcount == mlflowTrainingFileLimit):
            break
    
    
    np.set_printoptions(threshold=sys.maxsize)

    df_categories = df[df["ads_country_dst"].str.startswith(('1', '2', '3', '4', '5', '6', '7', '8', '9', '0')) == False] 
    df_categories = pd.concat([df_categories["ads_country_dst"]], axis=1, sort=False,)
    print("-------------- Count Record --------------")
    print(df_categories.shape[0])
    print("-------------- Count Record --------------")
    print("-------------- Count Destination Country --------------")
    print(df_categories.value_counts().to_string())
    print("-------------- Count Destination Country --------------")
    
    countryMap = mapOfCountryDst()
    print("-------------- Number of Country in Encoding --------------")
    print(len(countryMap))
    print("-------------- Number of Country in Encoding --------------")
    print("-------------- Show Country Not in list --------------")
    print(df_categories[~df_categories['ads_country_dst'].isin(countryMap.keys())].value_counts().to_string())
    print("-------------- Show Country Not in list --------------")

    df_categories = df_categories.mask(~df_categories.isin(countryMap.keys()),'OTHER')
    X = df_categories.replace({'ads_country_dst': countryMap})
    # print(X)
    # X_transform = createXTransformOrdinalDst()
    # X = X_transform.transform(df_categories)
    
    # Call and fit the Local Outlier Factor detector
    # setNNeighbors = int((df_categories.shape[0]/300)) This is best scenario but memory 64GB still OMM killed
    print("set n_neighbors : " , setNNeighbors)
    lof_detector = LocalOutlierFactor(n_neighbors=setNNeighbors, contamination=0.1,novelty=True).fit(X.values)
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
    print("-------------- List Destination Country with Prediction -------------")
    # print(type(df_categories.value_counts()))
    # print(type(df_categories))
    for index, value in df_categories.value_counts().items():
        encode = countryMap[index[0]]
        predictData = lof_detector.predict([[ encode ]])
        print(index[0]  , " | code =" , encode , " | count ="  , value , " | result =" , dataPredictionToString(predictData[0]))

    print("-------------- List Destination Country with Prediction -------------")
    
    plt.figure(figsize=(7,7))
    plt.scatter(X.to_numpy()[:, 0], X.to_numpy()[:, 0], c=lof_detect, cmap="flag", alpha=0.7)
    plt.title("train-ads-anomaly-dest-country")
    plt.savefig('train-ads-anomaly-dest-country.png')
    plt.show()

    tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
    # export MLFLOW_TRACKING_USERNAME=user 
    # export MLFLOW_TRACKING_PASSWORD=pwd
    experiment = mlflow.set_experiment(experiment_name='ads-anomaly-dest-country')
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
        
        mlflowMinioFolder
        mlflow.log_param("MlflowMinioFolder", mlflowMinioFolder)
        mlflow.log_param("CountryEncodingAmount", len(countryMap))
        mlflow.log_param("SampleFiles", xcount)
        mlflow.log_param("SampleRows", X.shape[0])

        # mlflow.set_tag("JenkinsURL",jenkinsURL)
        mlflow.log_metric("Anomaly", str((countDetect[0])*100/(countDetect[0]+countDetect[1])))
        mlflow.log_metric("Normally", str((countDetect[1])*100/(countDetect[0]+countDetect[1])))
        mlflow.sklearn.log_model(lof_detector, "model", registered_model_name="ads-anomaly-by-dest-country")
        print("Model saved in run %s" % mlflow.active_run().info.run_uuid)