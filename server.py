from flask import Flask, request, jsonify
import requests
import json
from waitress import serve
import os

import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sharelib import *

app = Flask(__name__)
# host = os.environ.get('host_ml', '127.0.0.1')
# port = os.environ.get('port_ml', '4567')
host = os.environ.get('host_ml', 'mlflow-app.rtarf-ml.its-software-services.com')
port = os.environ.get('port_ml', '80')

host_anomaly_des_country = os.environ.get('host_anomaly_des_country', 'mlflow-ads-anomaly-dest-country.rtarf-ml.its-software-services.com')
port_anomaly_des_country = os.environ.get('port_anomaly_des_country', '80')

gateway_port = os.environ.get('gateway_port_ml', '8082')

def createDataV2(request_country,request_timestamp):
    
    print(request_country + " " + request_timestamp)
    
    test_df = pd.DataFrame([[request_country,request_timestamp]],columns=['ads_country_dst', '@timestamp'])
    test_df = maskOfficeHour2(test_df)
    
    countryStr = listOfCountryDst()
    test_df['ads_country_dst'] = test_df['ads_country_dst'].mask(~test_df['ads_country_dst'].isin(countryStr),'OTHER')
    
    test_df = test_df.drop(['@timestamp'], axis=1)

    X_new = X_transform.transform(test_df)
    data = {
        "data":
        X_new.toarray().tolist()
    }

    return data

@app.route('/v2/gateway', methods=['POST'])
def get_invocationsV2():
    headers = {
        "Content-Type": "application/json",
    }

    content = request.json
    request_country = content['ads_country_dst']
    request_timestamp = content['@timestamp']
    content_data = createDataV2(request_country,request_timestamp)

    try:
        resp = requests.post(
            url="http://%s:%s/invocations" % (host, port),
            data=json.dumps({"dataframe_split": content_data}),
            headers=headers,
        )

        print(resp.status_code)
        return resp.json()

    except Exception as e:
        errmsg = "Caught exception attempting to call model endpoint: %s" % e
        print(errmsg, end="")
        return resp.json()

@app.route('/v3/gateway', methods=['GET'])
def get_MockData():

    mockData = {
                    "results": [
                        {
                            "subject": "supervised_dst_country_anomaly",
                            "result": "true",
                            "certainty": 0.99
                        },
                        {
                            "subject": "supervised_login_anomaly",
                            "result": "false",
                            "certainty": 1.00
                        },
                        {
                            "subject": "unsupervised_dst_country_anomaly",
                            "result": "Normally", #Anomaly
                            "certainty": 0.99
                        },
                        {
                            "subject": "unsupervised_login_anomaly",
                            "result": "Normally", #Anomaly
                            "certainty": 0.99
                        }
                    ]
                }

    jsonString = json.dumps(mockData, indent=4)
    
    return jsonString



def createDataAdsAnomalyDestCountry(request_country):
    
    test_df = pd.DataFrame([[request_country]],columns=['ads_country_dst'])
    
    test_df['ads_country_dst'] = test_df['ads_country_dst'].mask(~test_df['ads_country_dst'].isin(countryStr),'OTHER')
    X_new = X_transformDataAdsAnomalyDestCountry.transform(test_df)
    data = {
        "data":
        X_new.values.tolist()
    }

    return data

# This v4 gateway for 4 model serveing 
@app.route('/v4/gateway', methods=['POST'])
def get_invocationsV4():
    headers = {
        "Content-Type": "application/json",
    }

    predictionList = []
    content = request.json

    request_country = content['ads_country_dst']
    #if request from start at number refer to ip_address then retuen Normally
    content_data = createDataAdsAnomalyDestCountry(request_country)

    try:
        resp = requests.post(
            url="http://%s:%s/invocations" % (host_anomaly_des_country, port_anomaly_des_country),
            data=json.dumps({"dataframe_split": content_data}),
            headers=headers,
        )
        
        responseData = {
                            "subject": "unsupervised_dst_country_anomaly",
                            "result": dataPredictionToString(resp.json()["predictions"][0]) #Anomaly
                        }

        predictionList.append(responseData)
    except Exception as e:
        errmsg = "Caught exception attempting to call model endpoint: %s" % e
        print(errmsg, end="")
        return resp.json()

    
    responsePredictData = {"results": predictionList}
    jsonString = jsonify(responsePredictData)
    return jsonString

@app.route('/v4/country_count', methods=['GET'])
def getCountryCount():
    data = {"results": len(countryStr)}
    jsonString = json.dumps(data, indent=4)
    return jsonString
    
if __name__ == '__main__':
    # This pattern of obsolite Due to high startup time
    # df = pd.read_json("data/firewall-traffic.json", lines=True)
    # df_country = df["ads_country_dst"]
    # df_OfficeHour = maskOfficeHour2(df)
    # df_categories = pd.concat([df_country, df_OfficeHour['is_OfficeHour']], axis=1, sort=False,)
    # enc = OneHotEncoder(handle_unknown='ignore')
    # X_transform = make_column_transformer((enc,['ads_country_dst']),(enc,['is_OfficeHour']))
    # X_transform.fit(df_categories)

    # Make a Reverse Engineer Dataframe
    # initialize list of lists
    # data = [['Russian Federation', 'yes'], ['Russian Federation', 'no'], ['OTHER', 'yes'], ['OTHER', 'no']]    
    # df = pd.DataFrame(data, columns=['ads_country_dst', 'is_OfficeHour'])
    
    # enc = OneHotEncoder(handle_unknown='ignore')
    # X_transform = make_column_transformer((enc,['ads_country_dst']),(enc,['is_OfficeHour']))
    # X_transform.fit(df)
    # This pattern of obsolite Due to high startup time

    # load listOfCountryDst since server start    
    countryStr = listOfCountryDst()

    X_transformDataAdsAnomalyDestCountry = createXTransformOrdinalDst()
    X_transform = createXTransform()


    print("Server Ready On Port " + gateway_port)

    serve(app, host="0.0.0.0", port=gateway_port)
    