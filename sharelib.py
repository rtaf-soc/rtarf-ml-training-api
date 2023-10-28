import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
import sys

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
    ,'Brunei Darussalam','Tokelau','Martinique','Haiti','Mongolia','Bouvet Island'
    ,'Virgin Islands U','Rwanda','Donetsk','Northern Mariana Islands','Nicaragua','Taiwan'
    ,'Benin','Virgin Islands British','Burundi','Netherlands Antilles','Barbados','Belize','Andorra','Ethiopia','Tajikistan'
    ,'Holy See (Vatican City State)','Cayman Islands','French Guiana','Monaco','Somalia','Samoa','Saint Lucia'
    ,'Saint Kitts And Nevis','Niue','Chad','Marshall Islands','French Polynesia','Madagascar','Jersey'
    ,'Anonymous Proxy','Guyana','Guinea','Greenland','Gibraltar','Tuvalu'
    ,'Cape Verde','Maldives','Bhutan','Turkmenistan','Mayotte','Sao Tome And Principe','Sierra Leone']
    
    
    # print(len(countryStr))

    return countryStr


def mapOfCountryDst():
    countryStr = {
        'United States': 0
    , 'US': 0
    , 'Austria': 1
    , 'Thailand': 2
    , 'TH': 2
    , 'Brazil': 3
    , 'China': 4
    , 'Netherlands': 5
    , 'Singapore': 6
    , 'SG': 6
    , 'Japan': 7
    , 'Sweden': 8
    , 'United Kingdom': 9
    , 'Australia': 10
    , 'Asia Pacific Region': 11
    , 'Russian Federation': 12
    , 'Malaysia': 13
    , 'Korea Republic Of': 14
    , 'Hong Kong': 15
    , 'Canada': 16
    , 'Germany': 17
    , 'India': 18
    , 'France': 19
    , 'Italy': 20
    , 'Slovakia': 21
    , 'European Union': 22
    , 'Norway': 23
    , 'Ireland': 24
    , 'Indonesia': 25
    , 'Spain': 26
    , 'Israel': 27
    , 'Taiwan': 28
    , 'Switzerland': 29
    , 'South Africa': 30
    , 'Viet Nam': 31
    , 'Turkey': 32
    , 'Lao Peoples Democratic Republic': 33
    , 'Luxembourg': 34
    , 'Iran Islamic Republic Of': 35
    , 'Poland': 36
    , 'Belgium': 37
    , 'Mexico': 38
    , 'Finland': 39
    , 'Slovenia': 40
    , 'Argentina': 41
    , 'Uruguay': 42
    , 'Czech Republic': 43
    , 'Greece': 44
    , 'Ukraine': 45
    , 'Egypt': 46
    , 'Hungary': 47
    , 'Pakistan': 48
    , 'Colombia': 49
    , 'Saudi Arabia': 50
    , 'Romania': 51
    , 'Denmark': 52
    , 'Philippines': 53
    , 'Morocco': 54
    , 'Tokelau': 55
    , 'New Zealand': 56
    , 'Chile': 57
    , 'Moldova Republic Of': 58
    , 'Unknown': 59
    , 'Iraq': 60
    , 'Mauritius': 61
    , 'Mongolia': 62
    , 'Tunisia': 63
    , 'Bangladesh': 64
    , 'Bulgaria': 65
    , 'Venezuela Bolivarian Republic Of': 66
    , 'Portugal': 67
    , 'Cyprus': 68
    , 'Serbia': 69
    , 'United Arab Emirates': 70
    , 'Kenya': 71
    , 'Bahrain': 72
    , 'Cambodia': 73
    , 'Lithuania': 74
    , 'Kazakhstan': 75
    , 'Panama': 76
    , 'Nigeria': 77
    , 'Algeria': 78
    , 'Latvia': 79
    , 'Seychelles': 80
    , 'Ecuador': 81
    , 'Estonia': 82
    , 'Costa Rica': 83
    , 'Uzbekistan': 84
    , 'Croatia': 85
    , 'Iceland': 86
    , 'Palestinian Territory Occupied': 87
    , 'Barbados': 88
    , 'Myanmar': 89
    , 'Peru': 90
    , 'Sudan': 91
    , 'Kuwait': 92
    , 'Armenia': 93
    , 'Paraguay': 94
    , 'Belarus': 95
    , 'Puerto Rico': 96
    , 'Syrian Arab Republic': 97
    , 'Netherlands Antilles': 98
    , 'Dominican Republic': 99
    , 'Jamaica': 100
    , 'Qatar': 101
    , 'Georgia': 102
    , 'Virgin Islands British': 103
    , 'Nepal': 104
    , 'El Salvador': 105
    , 'CTe D Ivoire': 106
    , 'Zimbabwe': 107
    , 'Bosnia And Herzegovina': 108
    , 'Ghana': 109
    , 'Uganda': 110
    , 'Sri Lanka': 111
    , 'Trinidad And Tobago': 112
    , 'Tanzania United Republic Of': 113
    , 'Zambia': 114
    , 'Angola': 115
    , 'Macedonia The Former Yugoslav Republic Of': 116
    , 'Oman': 117
    , 'Malta': 118
    , 'Bouvet Island': 119
    , 'Bolivia Plurinational State Of': 120
    , 'Cape Verde': 121
    , 'Albania': 122
    , 'Cameroon': 123
    , 'Jordan': 124
    , 'Senegal': 125
    , 'Afghanistan': 126
    , 'Malawi': 127
    , 'RUnion': 128
    , 'Nicaragua': 129
    , 'Azerbaijan': 130
    , 'Gambia': 131
    , 'Honduras': 132
    , 'Belize': 133
    , 'Cayman Islands': 134
    , 'Ethiopia': 135
    , 'Congo The Democratic Republic Of The': 136
    , 'Congo': 136
    , 'Namibia': 137
    , 'Maldives': 138
    , 'Isle Of Man': 139
    , 'Guyana': 140
    , 'Mozambique': 141
    , 'Cuba': 142
    , 'Gabon': 143
    , 'Lebanon': 144
    , 'Crimea': 145
    , 'Libyan Arab Jamahiriya': 146
    , 'Togo': 147
    , 'Guatemala': 148
    , 'Guam': 149
    , 'Tajikistan': 150
    , 'Andorra': 151
    , 'Donetsk': 152
    , 'Turkmenistan': 153
    , 'Suriname': 154
    , 'Bhutan': 155
    , 'Bermuda': 156
    , 'Gibraltar': 157
    , 'Guadeloupe': 158
    , 'Kyrgyzstan': 159
    , 'Macao': 160
    , 'San Marino': 161
    , 'Holy See (Vatican City State)': 162
    , 'Sierra Leone': 163
    , 'Benin': 164
    , 'Tuvalu': 165
    , 'Brunei Darussalam': 166
    , 'Botswana': 167
    , 'Marshall Islands': 168
    , 'Mayotte': 169
    , 'Monaco': 170
    , 'Burkina Faso': 171
    , 'Bahamas': 172
    , 'Niue': 173
    , 'Aruba': 174
    , 'Antigua And Barbuda': 175
    , 'Anonymous Proxy': 176
    , 'Yemen': 177
    , 'Montenegro': 178
    , 'Papua New Guinea': 179
    , 'Chad': 180
    , 'Madagascar': 181
    , 'Guinea': 182
    , 'Liechtenstein': 183
    , 'Jersey': 184
    , 'Rwanda': 185
    , 'Djibouti': 186
    , 'Saint Kitts And Nevis': 187
    , 'Saint Lucia': 188
    , 'French Guiana': 189
    , 'Samoa': 190
    , 'French Polynesia': 191
    , 'Sao Tome And Principe': 192
    , 'Haiti': 193
    , 'Somalia': 194
    , 'New Caledonia': 195
    , 'Greenland': 196
    , 'Taiwan ROC': 197
    , 'Grenada': 198
    , 'Turks And Caicos Islands': 199
    , 'Kiribati': 200
    , 'Fiji': 201
    , 'Swaziland': 202
    , 'Land Islands': 203
    , 'Martinique': 204
    , 'Virgin Islands U': 205
    , 'Northern Mariana Islands': 206
    , 'Burundi': 207
    , 'Timor-Leste': 208
    , 'Liberia': 209
    , 'Mali': 210
    , 'Lesotho': 211
    , 'Curacao': 212
    , 'Equatorial Guinea': 213
    , 'Faroe Islands': 214
    , 'Niger': 215
    , 'Solomon Islands': 216
    , 'Sint Maarten (Dutch Part)': 217
    , 'Saint Martin (French Part)': 218
    , 'Bonaire Saint Eustatius And Saba': 219
    , 'Mauritania': 220
    , 'Central African Republic': 221
    , 'Guernsey': 222
    , 'Dominica': 223
    , 'Vanuatu': 224
    , 'OTHER': 250
    }

    return countryStr

def dataPredictionToString(predictCode):
    strMap = ""
    if (predictCode == 1):
        strMap = "Normally"
    else:
        strMap = "Anomaly"

    return strMap

def getArgs(n,defaultArg):
    strMap = ""
    maxArg = len(sys.argv)

    if (maxArg > n): 
        strMap = sys.argv[n]
    else: 
        strMap = defaultArg

    return strMap