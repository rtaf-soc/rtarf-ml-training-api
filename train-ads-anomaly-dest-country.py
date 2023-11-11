import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn import metrics

########### mflow ############
import mlflow
import mlflow.sklearn
########### mflow ############
from sharelib import *
from report import *
from pathlib import Path
import json
import glob
import sys
import pickle

jenkinsURL = getArgs(1,"")
mlflowMinioFolder = getArgs(2,"")
mlflowTrainingFileLimit = int(getArgs(3,10))
jenkinsBuildID = getArgs(4,"")

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

    # df_categories = df[df["ads_country_dst"].str.startswith(('1', '2', '3', '4', '5', '6', '7', '8', '9', '0')) == False] 
    df_categories = pd.concat([df["ads_country_dst"]], axis=1, sort=False,)
    print("-------------- Count Record --------------")
    print(df_categories.shape[0])
    print("-------------- Count Record --------------")
    print("-------------- Count Destination Country --------------")
    print(df_categories.value_counts().to_string())
    print("-------------- Count Destination Country --------------")
    
    countryMap = mapOfCountryDst()
    print("-------------- Number of Country in Encoding --------------")
    print("country_key : ", len(countryMap.keys()))
    print("country_count : ", len(set(countryMap.values())))
    print("-------------- Number of Country in Encoding --------------")
    print("-------------- Show Country Not in list --------------")
    print(df_categories[~df_categories['ads_country_dst'].isin(countryMap.keys())].value_counts().to_string())
    print("-------------- Show Country Not in list --------------")

    df_categories = df_categories.mask(~df_categories.isin(countryMap.keys()),'OTHER')
    print("Mask OTHER done")
    X = df_categories.replace({'ads_country_dst': countryMap})
    print("Frequency encoding done")
    
    normalPoint = 30
    X_Test = X.mask(X <= normalPoint, 1)
    X_Test.mask(X_Test > normalPoint, -1,inplace=True)
    
    # Call and fit the Local Outlier Factor detector
    setNNeighbors = int((df_categories.shape[0]/300)) # /30 This is best scenario but memory 64GB still OMM killed
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
    htmlAnomalyVSNormally = '''
    <table class="table table-striped">
        <th>Type</th><th class=right-aligned>Record</th><th class=right-aligned>%Record</th>
        <tr><td>Anomaly</td><td class=right-aligned>''' + str(countDetect[0]) + '''</td><td class=right-aligned> ''' + str((countDetect[0])*100/(countDetect[0]+countDetect[1])) + '''</td></tr>
        <tr><td>Normal</td><td class=right-aligned>''' + str(countDetect[1]) + '''</td><td class=right-aligned> ''' + str((countDetect[1])*100/(countDetect[0]+countDetect[1])) + '''</td></tr>
    </table>
    '''
    print("--------------Count Anomaly VS Normal-------------")
    print("-------------- List Destination Country with Prediction -------------")
    # print(type(df_categories.value_counts()))
    # print(type(df_categories))
    htmlItem = ""
    for index, value in df_categories.value_counts().items():
        encode = countryMap[index[0]]
        predictData = lof_detector.predict([[ encode ]])
        print(index[0]  , " | code =" , encode , " | count ="  , value , " | result =" , dataPredictionToString(predictData[0]))
        htmlItem = htmlItem + "<tr><td>" + index[0] + "</td><td class=right-aligned>" + str(encode) + "</td><td class=right-aligned>" + str(value)+ "</td><td>" + dataPredictionToString(predictData[0]) + "</td><tr>"
    
    htmlCountryPrediction = '''
    <table class="table table-striped">
        <th>Country</th><th class=right-aligned>CODE</th><th class=right-aligned>Amount</th><th>Prediction</th>
        ''' + htmlItem + '''
    </table>
    '''
    print("-------------- List Destination Country with Prediction -------------")
    
    plt.figure(figsize=(7,7))
    plt.scatter(X.to_numpy()[:, 0], X.to_numpy()[:, 0], c=lof_detect, cmap="flag", alpha=0.7)
    plt.title("train-ads-anomaly-dest-country")
    plt.savefig('images/train-ads-anomaly-dest-country.png')
    plt.show()

    print("-------------- Machine Learning - Confusion Matrix -------------")
    Accuracy = metrics.accuracy_score(X_Test, lof_detect)
    print("Accuracy : " , Accuracy)
    Precision = metrics.precision_score(X_Test, lof_detect)
    print("Precision : " , Precision)
    Sensitivity_recall = metrics.recall_score(X_Test, lof_detect)
    print("Sensitivity_recall : " , Sensitivity_recall)
    Specificity = metrics.recall_score(X_Test, lof_detect, pos_label=-1)
    print("Specificity : " , Specificity)
    F1_score = metrics.f1_score(X_Test, lof_detect)
    print("F1_score : " , F1_score)

    htmlMatrix = '''
    <table class="table table-striped">
        <th>Type</th><th>Meaning</th><th class=right-aligned>Score</th>
        <tr><td>Accuracy</td><td>The proportion of correctly predicted cases</td><td class=right-aligned>''' + str(Accuracy) + '''</td></tr>
        <tr><td>Precision</td><td>Positive Predictive Value</td><td class=right-aligned>''' + str(Precision) + '''</td></tr>
        <tr><td>Sensitivity_recall</td><td> True Positive Rate</td><td class=right-aligned>''' + str(Sensitivity_recall) + '''</td></tr>
        <tr><td>Specificity</td><td>True Negative Rate</td><td class=right-aligned>''' + str(Specificity) + '''</td></tr>
        <tr><td>F1_score</td><td>Balances precision and recall</td><td class=right-aligned>''' + str(F1_score) + '''</td></tr>    
    </table>
    '''
    print("-------------- Machine Learning - Confusion Matrix -------------")


    confusion_matrix = metrics.confusion_matrix(X_Test, lof_detect)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ["Anomally", "Normally"])
    cm_display.plot()
    plt.savefig('images/train-ads-anomaly-dest-country-confusion-matrix.png')
    plt.show()

    summary_table = '''
    <p>Count Record : ''' + str(df_categories.shape[0]) + '''</p>
    <h2>Local Outlier Factor (LOF)</h2>
    <p>n_neighbors : ''' + str(setNNeighbors) + '''</p>
    ''' + htmlAnomalyVSNormally + '''
    ''' + htmlCountryPrediction + '''
    <table class="table table-striped">
    <th>Local Outlier Factor (LOF)</th>
    <tr>
        <td><img src="train-ads-anomaly-dest-country.png" alt="train-ads-anomaly-dest-country.png"></td>
    </tr>
        </table>
    <table class="table table-striped">
    <th>confusion-matrix</th>
    <tr>
        <td><img src="train-ads-anomaly-dest-country-confusion-matrix.png" alt="confusion-matrix"></td>
    </tr>
    </table>
    ''' + htmlMatrix + '''
    '''

    html_string = mainReportHTML("train-ads-anomaly-dest-country",summary_table)   

    f = open('report.html','w')
    f.write(html_string)
    f.close()

    tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
    os.environ['MLFLOW_HTTP_REQUEST_TIMEOUT'] = '6000'
    # export MLFLOW_TRACKING_USERNAME=user 
    # export MLFLOW_TRACKING_PASSWORD=pwd
    experiment = mlflow.set_experiment(experiment_name='ads-anomaly-dest-country')
    experiment_id = experiment.experiment_id

    reportURL = "https://minio-api.rtarf-ml.its-software-services.com/ml-report/train-ads-anomaly-dest-country/" + jenkinsBuildID + "/report.html"

    run_description = f"""
### Note
**All information** * about Training * ~~ML here~~
Jenkins URL: [{jenkinsURL}]({jenkinsURL})
Report: [{reportURL}]({reportURL})
    """
    # mlflow.environment_variables.MLFLOW_ARTIFACT_UPLOAD_DOWNLOAD_TIMEOUT='6000'
    # mlflow.environment_variables.MLFLOW_HTTP_REQUEST_TIMEOUT='6000'

    # urllib3.util.retry.Retry(total=10, backoff_factor=0.1, status_forcelist=[ 500, 502, 503, 504 ])

    with mlflow.start_run(experiment_id=experiment_id,description=run_description):
        mlflow.set_tracking_uri(tracking_uri)

        print("Artifact Location: {}".format(experiment.artifact_location))
        print("artifact uri : " + mlflow.get_artifact_uri())

        mlflow.environment_variables.MLFLOW_ARTIFACT_UPLOAD_DOWNLOAD_TIMEOUT='6000'
        mlflow.environment_variables.MLFLOW_HTTP_REQUEST_TIMEOUT='6000'
        mlflow.doctor()
        
        mlflowMinioFolder
        mlflow.log_param("MlflowMinioFolder", mlflowMinioFolder)
        mlflow.log_param("country_key", len(countryMap.keys()))
        mlflow.log_param("country_count", len(set(countryMap.values())))
        mlflow.log_param("SampleFiles", xcount)
        mlflow.log_param("SampleRows", X.shape[0])

        # mlflow.set_tag("JenkinsURL",jenkinsURL)
        mlflow.log_metric("Anomaly", str((countDetect[0])*100/(countDetect[0]+countDetect[1])))
        mlflow.log_metric("Normally", str((countDetect[1])*100/(countDetect[0]+countDetect[1])))
        mlflow.sklearn.log_model(lof_detector, "model", registered_model_name="ads-anomaly-by-dest-country",await_registration_for=6000)
        print("Model saved in run %s" % mlflow.active_run().info.run_uuid)