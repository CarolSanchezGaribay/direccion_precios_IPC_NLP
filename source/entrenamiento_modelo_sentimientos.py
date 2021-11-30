import sys, os
sys.path.insert(0, os.getcwd()+'/source')
from analisis_descriptivo_tuits import (normalizar, tokenize, stop_words_esp,
                                        num_stopwords, prom_palabras)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix, auc, plot_roc_curve
from joblib import dump
import nltk
# from langdetect import detect

# =============================================================================
# INPUTS
# =============================================================================
INPUT_GT_TR = 'data/TASS_Dataset/general-train-tagged.csv'
INPUT_GT_TT = 'data/TASS_Dataset/general-test-tagged.csv'
INPUT_ST_TR = 'data/TASS_Dataset/stompol-train-tagged.csv'
INPUT_ST_TT = 'data/TASS_Dataset/stompol-test-tagged.csv'
INPUT_SV_TR = 'data/TASS_Dataset/socialtv-train-tagged.csv'
INPUT_SV_TT = 'data/TASS_Dataset/socialtv-test-tagged.csv'
INPUT_TRAINED_1 = 'data/tweets_data_trained.xlsx'
INPUT_TRAINED_2 = 'data/tweets_data_trained_2.xlsx'
INPUT_TRAINED_3 = 'data/tweets_data_trained_3.xlsx'

# =============================================================================
# Leer datos
# =============================================================================
general_tweets_corpus_train = pd.read_csv(INPUT_GT_TR, encoding='utf-8')
general_tweets_corpus_test = pd.read_csv(INPUT_GT_TT, encoding='utf-8')
stompol_tweets_corpus_train = pd.read_csv(INPUT_ST_TR, encoding='utf-8')
stompol_tweets_corpus_test = pd.read_csv(INPUT_ST_TT, encoding='utf-8')
social_tweets_corpus_train = pd.read_csv(INPUT_SV_TR, encoding='utf-8')
social_tweets_corpus_test = pd.read_csv(INPUT_SV_TT, encoding='utf-8')

tweets_corpus = pd.concat([
                           social_tweets_corpus_train,
                           social_tweets_corpus_test,
                           stompol_tweets_corpus_train,
                           stompol_tweets_corpus_test,
                           general_tweets_corpus_train,
                           general_tweets_corpus_test,
                          ])

tweets_corpus = tweets_corpus.query('agreement != "DISAGREEMENT" and polarity != "NONE"')
tweets_corpus['polarity'] = np.where(tweets_corpus['polarity']=='P+', 'P', tweets_corpus['polarity'])
tweets_corpus['polarity'] = np.where(tweets_corpus['polarity']=='N+', 'N', tweets_corpus['polarity'])
tweets_corpus = tweets_corpus[tweets_corpus.polarity != 'NEU']
#tweets_corpus['polarity_bin'] = tweets_corpus['polarity'].apply(lambda x: 1 if x in ['P', 'P+'] else -1)
#tweets_corpus_tt = tweets_corpus[(~tweets_corpus['content'].isna()) & (~tweets_corpus['polarity_bin'].isna())]
tweets_corpus_tt = tweets_corpus[~tweets_corpus['content'].isna()]
#tweets_corpus_tt = tweets_corpus_tt[['content', 'polarity_bin']]
tweets_corpus_tt = tweets_corpus_tt[['content', 'polarity']]

#Diferencia entre las clases "Positivo" y "Negativo"
tweets_corpus_tt['Num_Palabras'] = tweets_corpus_tt['content'].apply(lambda x: len(str(x).split(" ")))
tweets_corpus_tt['Num_Caraceteres'] = tweets_corpus_tt['content'].str.len()
tweets_corpus_tt['Num_Stopwords'] = tweets_corpus_tt['content'].apply(lambda x: num_stopwords(x))  
tweets_corpus_tt['Largo_promedio'] = tweets_corpus_tt['content'].apply(lambda x: prom_palabras(x))         
tweets_corpus_tt['Num_Hashtags'] = tweets_corpus_tt['content']\
    .apply(lambda x: len([x for x in str(x).split() if x.startswith('#')]))
tweets_corpus_tt['Num_Menciones'] = tweets_corpus_tt['content']\
    .apply(lambda x: len([x for x in str(x).split() if x.startswith('@')]))
tweets_corpus_tt['Num_Tickers'] = tweets_corpus_tt['content']\
    .apply(lambda x: len([x for x in str(x).split() if x.startswith('$')]))

# Tuits seleccionados aleatoriamente para clasificación y entrenamiento
"""
ruta1 = "data/TWEETS/TICKERS/"
ruta2 = "data/TWEETS/HASHTAGS/"
ruta3 = "data/TWEETS/INSTITUCIONES/"
ruta4 = "data/TWEETS/PERIODICOS/"
ruta5 = "data/TWEETS/PERSONAS/"
tickers, tipo_tick = carga_tweets(ruta1)
hashtags, tipo_hash = carga_tweets(ruta2)
instituciones, tipo_inst = carga_tweets(ruta3)
periodicos, tipo_perio = carga_tweets(ruta4)
personas, tipo_pers = carga_tweets(ruta5)
tipo_perio = [tipo[:-2] for tipo in tipo_perio if tipo[0]=='@']
tipo_perio = [tipo[:-1] if tipo in ['@El_Universal_Mx_', '@Milenio_'] else tipo for tipo in tipo_perio]
#data['TYPE'] = data['TYPE'].apply(lambda x: x[:-1] if x in ['@El_Universal_Mx_', '@Milenio_'] else x)
tipo_pers = [tipo[:-2] for tipo in tipo_pers if tipo[0]=='@']
datos = np.array(['usernameTweet','ID','text','url','nbr_retweet','nbr_faborite',
                  'nbr_reply','datetime','is_reply','is_retweet','user_id'])
raw_data = pd.DataFrame(tickers, columns=datos)
raw_data['TYPE'] = tipo_tick
for dicts, tipo in zip([hashtags, instituciones, periodicos, personas],
                        [tipo_hash, tipo_inst, tipo_perio, tipo_pers]):
    #print(tipo)
    data_dict = pd.DataFrame(dicts, columns=datos)
    data_dict['TYPE'] = tipo
    raw_data = pd.concat([raw_data, data_dict], axis=0, ignore_index=True)
    
#prueba = raw_data.groupby('TYPE', as_index=False).agg({'datetime': 'min'}).sort_values(by='datetime')
raw_data = raw_data.drop_duplicates(subset={'ID'})
data = raw_data[(raw_data.is_reply==False) & (raw_data.is_retweet==False)]
data.set_index('ID', inplace=True)
#data['datetime'] = pd.to_datetime(data['datetime'])
data['nbr_faborite'].fillna(0, inplace = True)
data = data.drop(['url','is_reply', 'is_retweet', 'user_id','nbr_faborite'], axis=1)
data['language'] = data['text'].apply(lambda x: detect(x))
data = data[data['language'].isin(['es', 'ca', 'pt', 'it', 'de','fr'])]
data = data[~data['text'].isna()]
data_a_calificar = data.sample(5000)
data_a_calificar.to_excel(r'Data/tweets_data_trained_2.xlsx', 
                        index=False,
                        encoding= 'latin-1')
"""

# =============================================================================
# DATA CALIFICADA MANUALMENTE CON LA DATA PROVEIDA POR EL TASS
# =============================================================================

trained_tweets_utf = pd.read_excel(INPUT_TRAINED_1)
trained_tweets_utf.columns = ['content', 'polarity']
trained_tweets_utf = trained_tweets_utf[~trained_tweets_utf['content'].isna()]
"""
trained_tweets_utf['Num_Palabras'] = trained_tweets_utf['content'].apply(lambda x: len(str(x).split(" ")))
trained_tweets_utf['Num_Caraceteres'] = trained_tweets_utf['content'].str.len()
trained_tweets_utf['Num_Stopwords'] = trained_tweets_utf['content'].apply(lambda x: num_stopwords(str(x))) 
trained_tweets_utf['Largo_promedio'] = trained_tweets_utf['content'].apply(lambda x: prom_palabras(str(x)))         
trained_tweets_utf['Num_Hashtags'] = trained_tweets_utf['content'].apply(lambda x: len([x for x in str(x).split() if x.startswith('#')]))      
trained_tweets_utf['Num_Menciones'] = trained_tweets_utf['content'].apply(lambda x: len([x for x in str(x).split() if x.startswith('@')]))
trained_tweets_utf['Num_Tickers'] = trained_tweets_utf['content'].apply(lambda x: len([x for x in str(x).split() if x.startswith('$')]))
"""

trained_tweets_utf_2 = pd.read_excel(INPUT_TRAINED_2)
trained_tweets_utf_2.columns = ['content', 'polarity']
trained_tweets_utf_2 = trained_tweets_utf_2[trained_tweets_utf_2['polarity']!='NEU']
trained_tweets_utf_2['polarity'] = trained_tweets_utf_2['polarity']\
    .apply(lambda x: 1 if x=='P' else 0)
"""
trained_tweets_utf_2['Num_Palabras'] = trained_tweets_utf_2['content'].apply(lambda x: len(str(x).split(" ")))
trained_tweets_utf_2['Num_Caraceteres'] = trained_tweets_utf_2['content'].str.len()
trained_tweets_utf_2['Num_Stopwords'] = trained_tweets_utf_2['content'].apply(lambda x: num_stopwords(str(x))) 
trained_tweets_utf_2['Largo_promedio'] = trained_tweets_utf_2['content'].apply(lambda x: prom_palabras(str(x)))         
trained_tweets_utf_2['Num_Hashtags'] = trained_tweets_utf_2['content'].apply(lambda x: len([x for x in str(x).split() if x.startswith('#')]))      
trained_tweets_utf_2['Num_Menciones'] = trained_tweets_utf_2['content'].apply(lambda x: len([x for x in str(x).split() if x.startswith('@')]))
trained_tweets_utf_2['Num_Tickers'] = trained_tweets_utf_2['content'].apply(lambda x: len([x for x in str(x).split() if x.startswith('$')]))
"""

trained_tweets_utf_3 = pd.read_excel(INPUT_TRAINED_3)
trained_tweets_utf_3.columns = ['content', 'polarity']
trained_tweets_utf_3 = trained_tweets_utf_3[trained_tweets_utf_3['polarity']!='NEU']
trained_tweets_utf_3['polarity'] = trained_tweets_utf_3['polarity']\
    .apply(lambda x: 1 if x=='P' else 0)
"""
trained_tweets_utf_3['Num_Palabras'] = trained_tweets_utf_3['content'].apply(lambda x: len(str(x).split(" ")))
trained_tweets_utf_3['Num_Caraceteres'] = trained_tweets_utf_3['content'].str.len()
trained_tweets_utf_3['Num_Stopwords'] = trained_tweets_utf_3['content'].apply(lambda x: num_stopwords(str(x))) 
trained_tweets_utf_3['Largo_promedio'] = trained_tweets_utf_3['content'].apply(lambda x: prom_palabras(str(x)))         
trained_tweets_utf_3['Num_Hashtags'] = trained_tweets_utf_3['content'].apply(lambda x: len([x for x in str(x).split() if x.startswith('#')]))      
trained_tweets_utf_3['Num_Menciones'] = trained_tweets_utf_3['content'].apply(lambda x: len([x for x in str(x).split() if x.startswith('@')]))
trained_tweets_utf_3['Num_Tickers'] = trained_tweets_utf_3['content'].apply(lambda x: len([x for x in str(x).split() if x.startswith('$')]))
"""

tweets_train = pd.concat([tweets_corpus_tt, trained_tweets_utf,
                          trained_tweets_utf_2, trained_tweets_utf_3],
                         axis=0, ignore_index=True)
tweets_train['polarity'] = tweets_train['polarity']\
    .apply(lambda x: 1 if x=='P' else 0)
tweets_train['content'] = tweets_train['content'].apply(lambda x: normalizar(x))

# =============================================================================
# Modelado en conjunto de entrenamiento
# =============================================================================

non_words = stop_words_esp()
vectorizer = TfidfVectorizer(
                analyzer = 'word',
                tokenizer = tokenize,
                lowercase = True,
                stop_words = non_words)
pipeline = Pipeline([
                    ('vect', vectorizer),
                    ('cls', LinearSVC()),
                    ])
X_train, X_test, y_train, y_test = train_test_split(tweets_train['content'], 
                                                    tweets_train['polarity'], 
                                                    test_size=0.1, 
                                                    random_state=42,
                                                    stratify=tweets_train['polarity'])
pipeline.fit(X_train, y_train)
# Búsqueda de mejores hiperparámetros
parameters = {
    'vect__max_df': (0.3, 0.5, 0.8),
    'vect__min_df': (10, 20, 50),
    'vect__max_features': (500, 750, 1000),
    'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    'cls__C': (0.2, 0.5, 0.7),
    'cls__loss': ('hinge', 'squared_hinge'),
    'cls__max_iter': (500, 1000)
}
rndm_search = RandomizedSearchCV(pipeline, parameters, n_jobs=-1 , random_state=0)
rndm_search.fit(X_train, y_train)
rndm_search.best_score_
columnas = rndm_search.best_estimator_.named_steps['vect'].get_feature_names()
best_model = rndm_search.best_estimator_
classifier = best_model.named_steps['cls']

#MUESTRA DE N-GRAMAS EN DATASET
sample_matrix = best_model.named_steps['vect'].transform(X_train)
sample_matrix_show = pd.DataFrame.sparse.from_spmatrix(sample_matrix, columns=columnas)
show_show = sample_matrix_show.merge(X_train.reset_index(drop=True), left_index=True, right_index=True, how='inner')
show_show.rename(columns = {'content': 'tweet'}, inplace=True)
show_show.set_index('tweet', inplace=True)
#show_show.drop('edurnity', axis=1, inplace=True)
show_show_show = show_show[(show_show['economico']>0) | (show_show['educacion']>0)]

# =============================================================================
# Prueba en tuits del sector políticp-económico-financiero
# =============================================================================
y_pred = best_model.predict(X_test)
#y_pred = pipeline.predict(X_test)
pruebas = pd.DataFrame({'y_test': y_test, 'y_pred': list(y_pred)})
prueba_final = pd.DataFrame(X_test).merge(pruebas, left_index=True, right_index=True, how='inner')
checar = prueba_final[(prueba_final['y_test'] == 1) &
                      (prueba_final['y_pred'] == 0)]
checar_2 = prueba_final[(prueba_final['y_test'] == 0) &
                        (prueba_final['y_pred'] == 1)]

# =============================================================================
# Métricas de desempeño del modelo
# =============================================================================
#MATRIZ DE CONFUSIÓN AL 0.5
confusion_matrix(y_test, y_pred)
#CURVA ROC
cv = StratifiedKFold(n_splits=5)
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
transformed_text = best_model.named_steps['vect'].transform(tweets_train['content'])
fig, ax = plt.subplots()
for i, (train, test) in enumerate(cv.split(transformed_text, tweets_train['polarity'])):
    classifier.fit(transformed_text[train], tweets_train['polarity'].iloc[train])
    viz = plot_roc_curve(classifier, transformed_text[test], tweets_train['polarity'].iloc[test],
                         name='Fold {}'.format(i+1),
                         alpha=0.3, lw=1, ax=ax)
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)
# Graficar ROC
ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Identidad', alpha=.8)
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(mean_fpr, mean_tpr, color='b',
        label=r'ROC promedio (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
        lw=2, alpha=.8)
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                label=r'$\pm$ 1 desv. est.')
ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
       title="ROC: Clasificación de sentimientos")
ax.legend(loc="lower right")
ax.set_aspect('equal')
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
#plt.show()
plt.savefig('images/roc_sentimientos.png')

# =============================================================================
# Exportar modelo
# =============================================================================
dump(best_model, 'utils/modelo_sentimientos.pkl')
