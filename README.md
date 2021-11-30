---
## Archivos de trabajo ##

### `source/analisis_descriptivo_simbolos.py` ###
Análisis descriptivo de los precios históricos de las acciones correspondientes a los símbolos que integran el sector financiero del BMX IPC (BBAJIO, BSMX, GENTERA, GFINBUR, GFNORT)

Input(s):
* data/supervised_tweets.csv
* data/STOCK_prices/BBAJIOO_2018_2020.csv
* data/STOCK_prices/BSMXB_2018_2020.csv
* data/STOCK_prices/GENTERA_2018_2020.csv
* data/STOCK_prices/GFINBURO_2018_2020.csv
* data/STOCK_prices/GFNORTEO_2018_2020.csv

Output(s): 
* images/precios_historicos.png
* images/variacion_historica.png
* images/matriz_corr_tickers.png
* images/hist_variacion_tickers.png
* images/hist_neg_variacion_tickers.png


### `source/analisis_descriptivo_tuits.py` ###

Análisis descriptivo de las opiniones publicadas en Twitter correspondientes a autores considerados influyentes en el sector político y/o económico y que tienen la cuenta verificada por la red social

Funciones de normalización de texto e ingeniería de variables de contexto

Input(s): 
* data/tweets_data_trained_2.xlsx
* data/tweets_data_trained_3.xlsx
* data/supervised_tweets.csv
* utils/lemma-es.csv

Output(s): 
* data/supervised_tweets_all_variables.csv
* images/palabrasfreq.png
* images/tuits_fechas.png
* images/tuits_horarios.png
* images/tuits_semana.png
* images/hist_num_palabras.png
* images/hist_num_caracteres.png
* images/hist_num_stopwords.png
* images/hist_larg_promedio.png
* images/hist_num_retweets.png
* images/hist_num_respuestas.png
* images/boxplots_sp_char_2.png
* images/boxplots_sp_char_3.png
* images/features_pairplot.png
* images/matriz_corr_tickers.png

### `entrenamiento_modelo_sentimientos.py` ###

Entrenamiento del modelo de sentimientos

Input(s):
* data/TASS_Dataset/general-train-tagged.csv
* data/TASS_Dataset/general-test-tagged.csv
* data/TASS_Dataset/stompol-train-tagged.csv
* data/TASS_Dataset/stompol-test-tagged.csv
* data/TASS_Dataset/socialtv-train-tagged.csv
* data/TASS_Dataset/socialtv-test-tagged.csv
* data/tweets_data_trained.xlsx
* data/tweets_data_trained_2.xlsx
* data/tweets_data_trained_3.xlsx

Output(s):
* utils/modelo_sentimientos.pkl
* images/roc_sentimientos.png

### `prediccion_modelo_sentimientos.py` ###

Predicción de variable *sentimiento* para todo el conjunto de datos

Input(s):
* utils/modelo_sentimientos.pkl
* data/TWEETS/HASHTAGS/
* data/TWEETS/INSTITUCIONES/
* data/TWEETS/PERIODICOS/
* data/TWEETS/PERSONAS/
* data/TWEETS/TICKERS/

Output(s):
* data/supervised_tweets.csv

### `entrenamiento_modelo_direccion_precios.py` ###

Input(s):
* data/supervised_tweets_all_variables.csv
* data/STOCK_prices/BBAJIOO_2018_2020.csv
* data/STOCK_prices/BSMXB_2018_2020.csv
* data/STOCK_prices/GENTERA_2018_2020.csv
* data/STOCK_prices/GFINBURO_2018_2020.csv
* data/STOCK_prices/GFNORTEO_2018_2020.csv

Output(s): 
* utils/modelo_direccion_precios.pkl
* images/curva_pr.png
* images/features_pairplot.png
* images/ts_cv.png
* images/balanceo_indicador_mercado.png
* images/norm_variables_01.png
* images/norm_variables_02.png
* images/norm_variables_03.png
* images/curva_aprendizaje.png 
* images/curva_aprendizaje_exactitud.png
* images/curva_aprendizaje_precision.png
* images/curva_validacion_c.png
* images/curva_validacion_c_acc.png
* images/curva_validacion_c_prec.png
* images/roc_oportunidades.png
* images/conf_matriz_0.png
* images/conf_matriz_05.png
* images/conf_matriz__05.png

---
## Data ##

`STOCK_prices` Carpeta de precios diarios de los tickers que pertenecen al sector financiero del IPC (BBAJIOO, BSMXB, GENTERA, GFINBURO, GFNORTEO)

`TASS_dataset` Tablas con tweets etiquetados para análisis de sentimientos 

`TWEETS` Carpeta con tweets que contengan los tickers $BBAJIO, $BSMX, $GENTERA, $GFINBU, $GFNORT y/o el hashtag #IPC

`tweets_data_trained.xlsx` 100 tweets clasificados

`tweets_data_trained_2.xlsx` 5094 tweets clasificados

`tweets_data_trained_3.xlsx` 5000 tweets clasificados

---
## Módulos importados ##

### TweetScraper ###

Clonar desde [aquí](https://github.com/jonbakerfish/TweetScraper/)

Contiene el módulo usado para scrapear los tuits de Twitter. 
Los pasos para usar esta biblioteca son:

1. Abrir la terminal
2. cd TweetScraper
3. scrapy list (deberá aparecer el tecto "TweetScrapper" en la pantalla del cmd)
4. Correr la consulta:
   * Opción 1: `scrapy crawl TweetScraper -a query="from:@lopezobrador_ since:2018-10-01 until:2018-12-31"`
   * Opción 2: `scrapy crawl TweetScraper -a query="$ GFNORT since:2018-10-01 until:2018-12-31" -a lang='es' -a crawl_user=true`

*Nota*: Las consultas usadas para el modelo de sentimientos se encuentran en el archivo `consultas_api_twitter.txt`   

### senti-py-master ###

Clonar desde [aquí](https://github.com/aylliote/senti-py)

Contiene el módulo para predecir el sentimiento de los tweets. El uso se encuentra en **predict_sentiments.py** sin embargo, no se usó debido a que preferí usar el dataset de TASS y crear mi propio predictor.
Los pasos para usar la biblioteca son:

1. Abrir Spyder
2. from classifier import *
3. clf = SentimentClassifier()
4. clf.predict(x) #donde x es texto
