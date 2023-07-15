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

# def eval_metrics(actual, pred):
#     rmse = np.sqrt(mean_squared_error(actual, pred))
#     mae = mean_absolute_error(actual, pred)
#     r2 = r2_score(actual, pred)
#     return rmse, mae, r2

if __name__ == "__main__":

    # df = pd.read_json("rawdata/test.txt", lines=True)
    # paths = Path("rawdata/").glob("*.txt")
    # df = pd.DataFrame([pd.read_json(p, lines=True) for p in paths])
    # df = pd.read_json("rawdata/ls.s3.b5b64958-2c75-40b0-b9d0-9a53a308aad5.2023-05-14T01.55.part200.txt", lines=True)

    df = pd.DataFrame()
    path_to_json = 'rawdata' 
    json_pattern = os.path.join(path_to_json,'*.txt')
    file_list = glob.glob(json_pattern)
    
    for file in file_list:
        data = pd.read_json(file, lines=True)
        df = pd.concat([df,data], ignore_index = True)
    
    # print("---------- data frame ---------")
    # print(df.head())
    # print(df.info())
    # print(df.describe())
    # print("---------- data frame ---------")
    
    # print("---------- @timestamp ---------")
    # print(df["@timestamp"])
    # print("---------- @timestamp ---------")
    
    df_country = df["ads_country_dst"]
    # print("---------- df_country ---------")
    # print(df_country.unique())
    # print(df_country.nunique())

    countryStr = listOfCountryDst()

    # countryStr = ['Russian Federation']
    df_country = df_country.mask(~df_country.isin(countryStr),'OTHER')
    # print(df_country.unique())
    # print(df_country.nunique())

    # print(df_country.value_counts())
    # Name: ads_country_dst, Length: 6679, dtype: object
    # Thailand                       3610
    # 10.0.0.0-10.255.255.255        1489
    # United States                  1031
    # Netherlands                      99
    # China                            67
    # Singapore                        54
    # 192.168.0.0-192.168.255.255      48
    # Hong Kong                        31
    # Korea Republic Of                29
    # Slovakia                         26
    # Asia Pacific Region              23
    # 100.64.0.0-100.127.255.255       22
    # Japan                            19
    # United Kingdom                   16
    # Spain                            16
    # Sweden                           15
    # Malaysia                         15
    # Australia                        13
    # Russian Federation               13
    # European Union                   10
    # Taiwan ROC                       10
    # Germany                           6
    # France                            4
    # Finland                           2
    # Poland                            2
    # 172.16.0.0-172.31.255.255         2
    # Denmark                           2
    # India                             1
    # Ireland                           1
    # Austria                           1
    # Brazil                            1
    # Canada                            1
    # Name: ads_country_dst, dtype: int64
    # print("---------- df_country ---------")

    df_OfficeHour = maskOfficeHour2(df)
    # print("---------- df_OfficeHour ---------")
    # print(df_OfficeHour['is_OfficeHour']) 
    # print(df_OfficeHour['is_OfficeHour'].value_counts()) 
    # # no     6678
    # # yes       1 
    # print("---------- df_OfficeHour ---------")

    
    # df['is_threat'] = pd.Series('no', index=df.index).mask(df['ads_country_dst']=="Russian Federation", 'yes')    
    df_threat = maskThreat2(df)
    # print("---------- Y ---------")
    print(df_threat['is_threat'].value_counts())    
    # # no     6667
    # # yes      12
    # # Name: is_threat, dtype: int64
    # print("---------- Y ---------")

    # df_OfficeHour['is_OfficeHour'] = df_OfficeHour['is_OfficeHour'].mask(df_OfficeHour['is_OfficeHour'] == "false","true")

    df_categories = pd.concat([df_country, df_OfficeHour['is_OfficeHour']], axis=1, sort=False,)
    
    # print("---------- df_categories ---------")
    # print(df_categories)   
    # print(df_categories.value_counts())

    # print(type(df_categories))
    # print(df_categories.describe)
    # print(df_categories.info())

    # print("---------- df_categories 1653 ---------")    
    # print(df_categories.iloc[498])
    # print(df_categories[df["ads_alert_by_dstip"] == "true"].to_string())
    # print(df_categories[df["ads_alert_by_dstip"] == "true"].value_counts())
#     ads_country_dst              is_OfficeHour
# 192.168.0.0-192.168.255.255  yes              188
# United States                yes               40
# Japan                        no                 8
# Japan                        yes                7
# United States                no                 1
    # print("---------- df_categories 1653 ---------")
    

    # ads_country_dst           is_OfficeHour
    # Thailand                     no               3610
    # 10.0.0.0-10.255.255.255      no               1489
    # United States                no               1031
    # Netherlands                  no                 99
    # China                        no                 67
    # Singapore                    no                 54
    # 192.168.0.0-192.168.255.255  no                 48
    # Hong Kong                    no                 31
    # Korea Republic Of            no                 29
    # Slovakia                     no                 26
    # Asia Pacific Region          no                 23
    # 100.64.0.0-100.127.255.255   no                 22
    # Japan                        no                 19
    # United Kingdom               no                 16
    # Spain                        no                 16
    # Sweden                       no                 15
    # Malaysia                     no                 15
    # Australia                    no                 13
    # Russian Federation           no                 12
    # European Union               no                 10
    # Taiwan ROC                   no                 10
    # Germany                      no                  6
    # France                       no                  4
    # Poland                       no                  2
    # Denmark                      no                  2
    # 172.16.0.0-172.31.255.255    no                  2
    # Finland                      no                  2
    # Russian Federation           yes                 1
    # Austria                      no                  1
    # Brazil                       no                  1
    # Canada                       no                  1
    # Ireland                      no                  1
    # India                        no                  1
    # dtype: int64   
    # print("---------- df_categories ---------")

    # print("---------- X_label ---------")
    # Create a LabelEncoder object and fit it to each feature in X
    # X_label = df_categories.apply(LabelEncoder().fit_transform)
    # print(X_label.value_counts())
    # ads_country_dst  is_OfficeHour
    # 29                  0                3610
    # 0                   0                1489
    # 31                  0                1031
    # 21                  0                  99
    # 9                   0                  67
    # 24                  0                  54
    # 3                   0                  48
    # 15                  0                  31
    # 19                  0                  29
    # 25                  0                  26
    # 4                   0                  23
    # 1                   0                  22
    # 18                  0                  19
    # 30                  0                  16
    # 26                  0                  16
    # 27                  0                  15
    # 20                  0                  15
    # 5                   0                  13
    # 23                  0                  12
    # 11                  0                  10
    # 28                  0                  10
    # 14                  0                   6
    # 13                  0                   4
    # 22                  0                   2
    # 10                  0                   2
    # 2                   0                   2
    # 12                  0                   2
    # 23                  1                   1
    # 6                   0                   1
    # 7                   0                   1
    # 8                   0                   1
    # 17                  0                   1
    # 16                  0                   1
    # dtype: int64
    # print("---------- X_label ---------")

    # print("---------- X ---------")
    # Create a OneHotEncoder object, and fit it to all of X
    # enc = OneHotEncoder(handle_unknown='ignore')
    # X_transform = make_column_transformer((enc,['ads_country_dst']),(enc,['is_OfficeHour']))
    # X_transform.fit(df_categories)
    X_transform = createXTransform()
    X = X_transform.transform(df_categories)
    # print("---------- X ---------")
    
    # print("---------- y ---------")    
    # y = df_threat['is_threat']
    y = df["ads_alert_by_dstip"]
    y = y.mask(y.isna(),"false")
    print(y.value_counts())
    print(y.loc[y == "true"])
    # print(X[498])
    # print("---------- x.loc[y == true] ---------")    
    print(X[y == "true"])
    # print("---------- x.loc[y == true] ---------")   
    # 1653    yes
    # 1886    yes
    # 2232    yes
    # 2783    yes
    # 3575    yes
    # 3646    yes
    # 4099    yes
    # 4580    yes
    # 4701    yes
    # 4758    yes
    # 4900    yes
    # 5495    yes
    # print("---------- y ---------")

    print("---------------------")
    # # Split the data into train and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1,train_size=0.9)
    print("X_train shape is:", X_train.shape)
    print("y_train shape is:", y_train.shape)
    print("X_test shape is:", X_test.shape)
    print("y_test shape is:", y_test.shape)


    # # Train the model
    logreg = LogisticRegression(max_iter=20000, random_state=42).fit(X_train, y_train)

    # # Evaluate the model's accuracy
    score_train = logreg.score(X_train, y_train)
    score_test = logreg.score(X_test, y_test)
    print("Train set accuracy = " + str(score_train))
    print("Test set accuracy = " + str(score_test))

    tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
    # export MLFLOW_TRACKING_USERNAME=user 
    # export MLFLOW_TRACKING_PASSWORD=pwd

    # experiment_id = mlflow.create_experiment("soc-ml-api2")
    experiment = mlflow.get_experiment_by_name('soc-ml-default')
    experiment_id = experiment.experiment_id

    with mlflow.start_run(experiment_id=experiment_id):
        mlflow.set_tracking_uri(tracking_uri)
        
        # experiment = mlflow.get_experiment_by_name('soc-ml-api2')
        print("Artifact Location: {}".format(experiment.artifact_location))
        print("artifact uri : " + mlflow.get_artifact_uri())

        mlflow.doctor()

        mlflow.log_metric("score_trains", str(score_train))
        mlflow.log_metric("score_test", str(score_test))
        mlflow.sklearn.log_model(logreg, "model", registered_model_name="soc-ml")
        print("Model saved in run %s" % mlflow.active_run().info.run_uuid)


    # print("----------- Make a prediction ------------")
    # print(X.shape[0])
    # # print(type(X))
    # count = 0
    # for icount in range(X.shape[0]):
    #     X_new = X[icount].toarray().tolist()
    #     # print(X_new)

    #     y_pred = logreg.predict(X_new)
    #     if y_pred[0] == "true":
    #         print("Predicted target true:", icount)
    #         count = count+1
    # #     # else:
    # #     #     print("icount: ", icount)

    # print("count true:", count)

    # X_new = X[498].toarray().tolist()
    # X_new = X[498]
    # y_pred = logreg.predict(X_new)
    # print("Predicted target name:", y_pred[0])

    # # Make a prediction
    # test_df = pd.DataFrame([
    # ['Russian Federation','Feb 22, 2023 @ 17:59:59.942']
    # ])
    # test_df.columns = ["ads_country_dst","@timestamp"]
    # test_df = maskOfficeHour(test_df)
    # test_df = test_df.drop(['@timestamp'], axis=1)
    # print(test_df)
    # print(test_df.info())

    # # X_transform fron all data fit
    # print(X_transform)
    # X_new = X_transform.transform(test_df)
    # print("X_new : " , X_new)

    # y_pred = logreg.predict(X_new)
    # y_pred_prob = logreg.predict_proba(X_new)
    # print("Prediction:", y_pred, "with the probability array:", y_pred_prob)
    # print("Predicted target name:", y_pred[0])



    # print("----------- Predict ------------")