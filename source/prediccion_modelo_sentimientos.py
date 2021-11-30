import sys, os
sys.path.insert(0, os.getcwd()+'/source')
from analisis_descriptivo_tuits import normalizar, carga_tweets
import numpy as np
import pandas as pd
import datetime
# from classifier import *
from joblib import load
# from langdetect import detect

# =============================================================================
# Inputs
# =============================================================================

ruta1 = "data/TWEETS/HASHTAGS/"
ruta2 = "data/TWEETS/INSTITUCIONES/"
ruta3 = "data/TWEETS/PERIODICOS/"
ruta4 = "data/TWEETS/PERSONAS/"
ruta5 = "data/TWEETS/TICKERS/"
MODELO_SENTIMIENTOS = "utils/modelo_sentimientos.pkl"

# =============================================================================
# Leer datos
# =============================================================================

hashtags, hashtags_query = carga_tweets(ruta1)
instituciones, instituciones_query = carga_tweets(ruta2)
periodicos, periodicos_query = carga_tweets(ruta3)
personas, personas_query = carga_tweets(ruta4)
tickers, tickers_query = carga_tweets(ruta5)
# Nombres de las columnas
datos = np.array(['usernameTweet', 'ID', 'text', 'url',
                  'nbr_retweet', 'nbr_faborite',
                  'nbr_reply', 'datetime', 'is_reply',
                  'is_retweet', 'user_id'])
# Reunir todos los dataframes
raw_data = pd.DataFrame(hashtags, columns=datos)
for dicts in [instituciones, periodicos, personas, tickers]:
    data_dict = pd.DataFrame(dicts, columns=datos)
    raw_data = pd.concat([raw_data, data_dict], axis=0, ignore_index=True)
raw_data = raw_data.drop_duplicates(subset={'ID'})
data = raw_data[(raw_data["is_reply"] == False) &
                (raw_data["is_retweet"] == False)]
data.set_index('ID', inplace=True)
#data['datetime'] = pd.to_datetime(data['datetime'])
data['nbr_faborite'].fillna(0, inplace=True)
data = data.drop(['url', 'is_reply', 'is_retweet', 'user_id', 'nbr_faborite'],
                 axis=1)
# data['language'] = data['text'].apply(lambda x: detect(x))
# Filtrar los tuits cuyo idioma sea en español o uno parecido
# data = data[data['language'].isin(['es', 'ca', 'pt', 'it', 'de','fr'])]
data = data[~data['text'].isna()]
data['datetime'] = pd.to_datetime(data['datetime'], format="%Y-%m-%d %H:%M:%S")
data = data[(data['datetime'] >= datetime.datetime(2018, 10, 1)) &
            (data['datetime'] <= datetime.datetime(2019, 12, 31))]\
            .sort_values(by='datetime')
data_to_predict = data['text'].apply(lambda x: normalizar(x))

# =============================================================================
# Predicciones
# =============================================================================

modelo_senti = load(MODELO_SENTIMIENTOS)
data['Sentimiento'] = modelo_senti.predict(data_to_predict)
'''
clf = SentimentClassifier()
data['polarity'] = data['text'].apply(lambda x: clf.predict(x))
'''

# =============================================================================
# Exportar datos
# =============================================================================
data.to_csv('data/supervised_tweets.csv',
            sep=',', 
            index=False, 
            encoding='utf-8')
