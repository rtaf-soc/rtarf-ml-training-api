import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder

def maskThreat(df):
    toThreatHourStr = '08:00:00.000'
    fromThreatHourStr = '17:00:00.000'
    # countryStr = ["Russian Federation","Slovakia","China","Netherlands","Thailand"]
    countryStr = ["Russian Federation"]
    return df.assign(is_threat=pd.Series('no', index=df.index).mask(((df['ads_country_dst'].isin(countryStr)) & ((df['@timestamp'].str[-12:]<toThreatHourStr) | (df['@timestamp'].str[-12:]>fromThreatHourStr))), 'yes'))

def maskOfficeHour(df):
    fromHourStr = '08:00:00.000'
    toHourStr = '17:00:00.000'
    return df.assign(is_OfficeHour=pd.Series('no', index=df.index).mask((((df['@timestamp'].str[-12:]>=fromHourStr) & (df['@timestamp'].str[-12:]<=toHourStr))), 'yes'))

def maskThreat2(df):
    toThreatHourStr = '08:00:00'
    fromThreatHourStr = '17:00:00'
    # countryStr = ["Russian Federation","Slovakia","China","Netherlands","Thailand"]
    countryStr = ["Russian Federation"]
    return df.assign(is_threat=pd.Series('false', index=df.index).mask(((df['ads_country_dst'].isin(countryStr)) & ((df['@timestamp'].str[11:19]<toThreatHourStr) | (df['@timestamp'].str[11:19]>fromThreatHourStr))), 'true'))

def maskOfficeHour2(df):
    fromHourStr = '08:00:00'
    toHourStr = '17:00:00'
    return df.assign(is_OfficeHour=pd.Series('no', index=df.index).mask((((df['@timestamp'].str[11:19]>=fromHourStr) & (df['@timestamp'].str[11:19]<=toHourStr))), 'yes'))

def maskThreat3(df):
    return df.assign(is_threat=pd.Series('false', index=df.index).mask(((df['ads_alert_by_dstip']=="true") | (df['ads_alert_by_blacklist_dstip']=="true")), 'true'))

def createXTransform():

    temp_df = pd.DataFrame(listOfCountryDst(), columns=['ads_country_dst'])
    temp_df = temp_df.assign(is_OfficeHour='yes')
    temp_df.loc[len(temp_df.index)] = ['OTHER', 'no'] 

    enc = OneHotEncoder(handle_unknown='ignore')
    X_transform = make_column_transformer((enc,['ads_country_dst']),(enc,['is_OfficeHour']))
    X_transform.fit(temp_df)

    return X_transform

def createXTransformDst():

    temp_df = pd.DataFrame(listOfCountryDst(), columns=['ads_country_dst'])
    temp_df.loc[len(temp_df.index)] = ['OTHER'] 

    enc = OneHotEncoder(handle_unknown='ignore')
    X_transform = make_column_transformer((enc,['ads_country_dst']))
    X_transform.fit(temp_df)

    return X_transform

def createXTransformTime():

    temp_df = pd.DataFrame(['yes','no'], columns=['is_OfficeHour'])
    
    enc = OneHotEncoder(handle_unknown='ignore')
    X_transform = make_column_transformer((enc,['is_OfficeHour']))
    X_transform.fit(temp_df)

    return X_transform

def createXTransformOrdinalDst():

    temp_df = pd.DataFrame(listOfCountryDst(), columns=['ads_country_dst'])
    temp_df.loc[len(temp_df.index)] = ['OTHER']

    scaler = OrdinalEncoder().set_output(transform="pandas")
    # scaler = OrdinalEncoder()
    X_transform = scaler.fit(temp_df)
    
    return X_transform

def listOfCountryDst():

    countryStr = ['United States','Thailand','10.0.0.0-10.255.255.255','Singapore','Kenya'
    ,'Spain','Japan','Netherlands','China','192.168.0.0-192.168.255.255'
    ,'France','Colombia','European Union','Sweden','Germany','Taiwan ROC'
    ,'Saudi Arabia','India','Canada','Denmark','Italy','Brazil','Indonesia'
    ,'Finland','100.64.0.0-100.127.255.255','Hong Kong','Austria','Tunisia'
    ,'172.16.0.0-172.31.255.255','Portugal','Russian Federation','Pakistan'
    ,'Korea Republic Of','Belgium','United Kingdom','Asia Pacific Region'
    ,'Argentina','South Africa','Australia','Bulgaria','Morocco','Ireland'
    ,'Greece','Azerbaijan','Hungary','Venezuela Bolivarian Republic Of'
    ,'United Arab Emirates','Norway','Turkey','Slovakia','Switzerland'
    ,'Mexico','Viet Nam','CTe D Ivoire','Puerto Rico','Malaysia','Chile'
    ,'Czech Republic','Poland','Cyprus','Slovenia','Egypt'
    ,'Iran Islamic Republic Of','Ukraine','Armenia','Cameroon','El Salvador'
    ,'Mauritius','Iraq','Ghana','Malawi','Croatia','Kazakhstan','Panama'
    ,'Unknown','Philippines','Estonia','Grenada','Israel','Nigeria'
    ,'Lithuania','Romania','San Marino','New Caledonia','Costa Rica','Latvia'
    ,'Zambia','New Zealand','Dominican Republic','Moldova Republic Of','Peru'
    ,'Algeria','Iceland','Luxembourg','Turks And Caicos Islands'
    ,'169.254.0.0-169.254.255.255','Ecuador','Kyrgyzstan','Kiribati','Belarus'
    ,'Serbia','Bermuda','Bolivia Plurinational State Of','Uruguay'
    ,'Trinidad And Tobago','Antigua And Barbuda','Fiji','Bangladesh'
    ,'Palestinian Territory Occupied','Swaziland','Syrian Arab Republic'
    ,'Sri Lanka','Honduras','Tanzania United Republic Of'
    ,'Lao Peoples Democratic Republic','Qatar','Suriname','Macao','Malta'
    ,'Liechtenstein','Lebanon','Nepal','Libyan Arab Jamahiriya','Cuba'
    ,'Gambia','Bosnia And Herzegovina','Namibia','Georgia','Sudan','Jordan'
    ,'Bahrain','Macedonia The Former Yugoslav Republic Of','Kuwait','Bahamas'
    ,'Paraguay','Uganda','Cambodia','Botswana','Guatemala','Oman'
    ,'Burkina Faso','Gabon','Angola','Land Islands','Papua New Guinea'
    ,'Afghanistan','Crimea','Isle Of Man','Zimbabwe','Togo','Guam','Yemen'
    ,'Mozambique','Congo The Democratic Republic Of The','Seychelles','Aruba'
    ,'Djibouti','Myanmar','Senegal'
    ,'Montenegro','Guadeloupe','Uzbekistan','Albania','RUnion','Jamaica'
    ,'Brunei Darussalam']

    # print(len(countryStr))

    return countryStr