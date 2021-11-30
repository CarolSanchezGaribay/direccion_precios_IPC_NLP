import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from datetime import datetime
from sklearn.svm import LinearSVC, SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import scale
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, \
    learning_curve, cross_validate, validation_curve
from sklearn.metrics import confusion_matrix, plot_roc_curve, \
    auc, average_precision_score, precision_recall_curve, \
    classification_report, f1_score
from joblib import dump
import warnings
warnings.filterwarnings("ignore")
print("Directorio actual: " + os.getcwd())

# =============================================================================
# INPUTS
# =============================================================================
INPUT_TUITS = 'data/supervised_tweets_all_variables.csv'
INPUT_BBAJIOO = 'data/STOCK_prices/BBAJIOO_2018_2020.csv'
INPUT_BSMXB = 'data/STOCK_prices/BSMXB_2018_2020.csv'
INPUT_GENTERA = 'data/STOCK_prices/GENTERA_2018_2020.csv'
INPUT_GFINBURO = 'data/STOCK_prices/GFINBURO_2018_2020.csv'
INPUT_GFNORTEO = 'data/STOCK_prices/GFNORTEO_2018_2020.csv'

# =============================================================================
# FUNCIONES
# =============================================================================

# CURVA DE APRENDIZAJE


def plot_learning_curve(estimator, X, y, title="Curva de aprendizaje",
                        axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.5, .625, 1)):
    if axes is None:
        _, axes = plt.subplots(1, 1, figsize=(20, 5))
    axes.set_title(title)
    if ylim is not None:
        axes.set_ylim(*ylim)
    axes.set_xlabel("Training examples")
    axes.set_ylabel("Score")
    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    # Plot learning curve
    axes.grid()
    axes.fill_between(train_sizes, train_scores_mean - train_scores_std,
                      train_scores_mean + train_scores_std, alpha=0.1,
                      color="r")
    axes.fill_between(train_sizes, test_scores_mean - test_scores_std,
                      test_scores_mean + test_scores_std, alpha=0.1,
                      color="g")
    axes.plot(train_sizes, train_scores_mean, 'o-', color="r",
              label="Training score")
    axes.plot(train_sizes, test_scores_mean, 'o-', color="g",
              label="Cross-validation score")
    axes.legend(loc="best")
    return plt


# VALIDACIÓN CRUZADA EN SERIE DE TIEMPO


def plot_cv_indices(cv, X, y, group, ax, lw=20, cmap_cv='RdBu_r', cmap_data='Blues_r'):
    """Create a sample plot for indices of a cross-validation object."""
    n_splits = cv.n_splits
    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(cv.split(X=X, y=y, groups=group)):
        # Fill in indices with the training/test groups
        print(tr, tt)
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0
        # Visualizar los resultados
        ax.scatter(range(len(indices)), [ii + .5] * len(indices),
                   c=indices, marker='_', lw=lw, cmap=cmap_cv,
                   vmin=-.2, vmax=1.2)
    # Graficar las clases de datos y grupos al final
    ax.scatter(range(len(X)), [ii + 1.5] * len(X),
               c=y, marker='_', lw=lw, cmap=cmap_data)
    # Dar formato
    yticklabels = list(range(1, n_splits + 1)) + ['Indicador']
    # xticklabels = X.index
    ax.set(yticks=np.arange(n_splits + 2) + .5, yticklabels=yticklabels,
           # xticklabels = xticklabels,
           xlabel='Indice', ylabel="Iteración CV",
           ylim=[n_splits + 1.1, -.2], xlim=[0, 250])
    ax.set_title('Validación Cruzada en Serie de Tiempo', fontsize=12)
    ax.tick_params(labelsize=8)
    # plt.xticks(rotation=70)
    plt.yticks(rotation=90, verticalalignment="center")
    return ax


# CURVA PRECISIÓN-RECALL


def draw_cv_pr_curve(classifier, cv, X, y, title='PR Curve'):
    y_real = []
    y_proba = []
    i = 0
    plt.subplots(figsize=(8, 8))
    for train, test in cv.split(X, y):
        probas_ = classifier.fit(X[train], y.iloc[train]).decision_function(X[test])
        # Calcular la curva ROC y el área
        precision, recall, _ = precision_recall_curve(y.iloc[test], probas_)
        # Graficar cada curva PR individualmente
        plt.plot(recall, precision, lw=1, alpha=0.7,
                 label='PR iteración %d (AUC = %0.2f)' %
                       (i + 1, average_precision_score(y.iloc[test], probas_)))
        y_real.append(y.iloc[test])
        y_proba.append(probas_)
        i += 1
    y_real = np.concatenate(y_real)
    y_proba = np.concatenate(y_proba)
    precision, recall, _ = precision_recall_curve(y_real, y_proba)
    plt.plot(recall, precision, color='b',
             label=r'PR promedio (AUC = %0.2f)' %
                   (average_precision_score(y_real, y_proba)),
             lw=2, alpha=.8)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Sensibilidad', fontsize=15)
    plt.ylabel('Precisión', fontsize=15)
    plt.title(title, fontsize=20)
    plt.legend(loc='best', fontsize=15)
    plt.show()
    # plt.savefig(r'images/curva_pr.png')


# =============================================================================
#if __name__ == "__main__":
tweets_data = pd.read_csv(INPUT_TUITS,
                          sep=',', encoding='utf-8')
data = tweets_data.drop(['usernameTweet',
                         'date', 'time',
                         'weekday'], axis=1)
data = data[~data['text'].isna()]
data['datetime'] = pd.to_datetime(data['datetime'], errors='coerce')
data['Fecha'] = data['datetime'].dt.date
data = data.drop(['datetime'], axis=1)
data['time_2'] = data['time_2'].str.slice(0, 2)
# data['language'] = data['text'].apply(lambda x: detect(x))
# data = data[data['language'].isin(['es', 'ca', 'pt', 'it', 'de','fr'])]
data = data[data['Sentimiento'].isin(['P', 'N'])]
data['Sentimiento'] = data['Sentimiento'].apply(lambda x: 1 if x == 'P' else -1)
# DATOS AGRUPADOS
data_gr_time = data.pivot_table(index='Fecha', columns='time_2',
                                values='text', aggfunc='count', fill_value=0)
data_grouped = data.groupby('Fecha').agg({'nbr_retweet': ['sum', 'mean'],
                                          'nbr_reply': ['sum', 'mean'],
                                          'Sentimiento': 'sum',
                                          'Num_Palabras': ['sum', 'mean'],
                                          'Num_Caraceteres': ['sum', 'mean'],
                                          'Num_Stopwords': ['sum', 'mean'],
                                          'Num_Hashtags': ['sum', 'mean'],
                                          'Num_Menciones': ['sum', 'mean'],
                                          'Num_Tickers': ['sum', 'mean'],
                                          'text': 'count'})
data_grouped.columns = [(col_1 + '_' + col_2) for (col_1, col_2) in data_grouped.columns]
data_grouped = data_grouped.merge(data_gr_time, left_index=True, right_index=True)
data_grouped.reset_index(inplace=True)
# SIMBOLOS
BBAJIO = pd.read_csv(INPUT_BBAJIOO)
BSMX = pd.read_csv(INPUT_BSMXB)
GENTERA = pd.read_csv(INPUT_GENTERA)
GFINBU = pd.read_csv(INPUT_GFINBURO)
GFNORT = pd.read_csv(INPUT_GFNORTEO)
tickers_dict = {'BBAJIO': BBAJIO,
                'BSMX': BSMX,
                'GENTERA': GENTERA,
                'GFINBU': GFINBU,
                'GFNORT': GFNORT}
for ticker in tickers_dict.keys():
    ind_dec = tickers_dict[ticker]['Apertura'] - tickers_dict[ticker]['Cierre']
    tickers_dict[ticker]['ticker'] = str(ticker)
    tickers_dict[ticker]['INDICATOR'] = ind_dec.apply(lambda x: -1 if x > 0 else 1)
    tickers_dict[ticker]['Fecha'] = pd.to_datetime(tickers_dict[ticker]['Fecha'],
                                                   errors='coerce',
                                                   format='%d.%m.%Y')
    tickers_dict[ticker]['Fecha'] = tickers_dict[ticker]['Fecha'].dt.date
    # tickers_dict[ticker]['Volumen'] = tickers_dict[ticker]['Vol.'].apply(lambda x: values(x))
    tickers_dict[ticker]['Variacion(%)'] = tickers_dict[ticker]['% var.'].apply(lambda x: float(x[:-1]))
    # tickers_dict[ticker].drop(['Vol.', '% var.'], axis=1, inplace=True)
    tickers_dict[ticker] = tickers_dict[ticker] \
        .sort_values(by=['Fecha']) \
        .reset_index(drop=True)
tickers_data = pd.DataFrame()
for ticker in tickers_dict.keys():
    tickers_data = pd.concat([tickers_data, tickers_dict[ticker]],
                             axis=0, sort=False, ignore_index=True)
tickers_data['Fecha'] = pd.to_datetime(tickers_data['Fecha'])
tickers_data = tickers_data[tickers_data['Fecha'].between(datetime(2018, 10, 1),
                                                          datetime(2019, 9, 30))]
dias_actividad = list(tickers_dict['BBAJIO']['Fecha'].unique())

# INDICADOR DE COMPRA
df_dias_actividad = pd.DataFrame({'Fecha': dias_actividad})
df_dias_actividad.reset_index(inplace=True)
poc_ant = tickers_data.groupby('Fecha', as_index=False).agg({'Cierre': 'sum'})
poc_ant_copy = poc_ant.copy()
poc_ant['Fecha_Ant'] = ''
for i, fecha in enumerate(poc_ant['Fecha']):
    indice = df_dias_actividad[df_dias_actividad['Fecha'] == fecha]['index'].iloc[0]
    poc_ant['Fecha_Ant'].iloc[i] = df_dias_actividad[df_dias_actividad['index'] == indice - 1]['Fecha'].iloc[0]
poc_ant['Fecha_Ant'] = pd.to_datetime(poc_ant['Fecha_Ant'])
poc = poc_ant.merge(poc_ant_copy, how='left', left_on='Fecha_Ant', right_on='Fecha')
poc.drop(['Fecha_y'], axis=1, inplace=True)
poc.rename(columns={'Fecha_x': 'Fecha_Actual',
                    'Cierre_x': 'Cierre_Actual',
                    'Cierre_y': 'Cierre_Ant'}, inplace=True)
del poc_ant, poc_ant_copy, df_dias_actividad, fecha, i, indice
poc['Variacion'] = (poc['Cierre_Actual'] / poc['Cierre_Ant']) - 1
poc['Indicator'] = np.where(poc['Variacion'] < 0.003, 0, 1)
poc['Rolling_mean_10'] = poc['Cierre_Actual'].rolling(10).mean()
poc['Rolling_mean_5'] = poc['Cierre_Actual'].rolling(5).mean()
poc['Rolling_mean_2'] = poc['Cierre_Actual'].rolling(2).mean()
poc['Rolling_std_10'] = poc['Cierre_Actual'].rolling(10).std()
poc['Rolling_std_5'] = poc['Cierre_Actual'].rolling(5).std()
poc['Rolling_std_2'] = poc['Cierre_Actual'].rolling(2).std()
for col, size in zip(['Rolling_mean_10', 'Rolling_mean_5', 'Rolling_mean_2'],
                     [10, 5, 2]):
    poc[col] = np.where(poc[col].astype(str) == 'nan',
                        poc['Cierre_Actual'].iloc[0:size].mean(), poc[col])
for col, size in zip(['Rolling_std_10', 'Rolling_std_5', 'Rolling_std_2'],
                     [10, 5, 2]):
    poc[col] = np.where(poc[col].astype(str) == 'nan',
                        poc['Cierre_Actual'].iloc[0:size].std(), poc[col])
# REUNIR DATOS FINANCIEROS CON OPINIONES
data_grouped['Fecha'] = pd.to_datetime(data_grouped['Fecha'])
# Alinear el indicador a los tuits del día anterior
data_input = poc.merge(data_grouped, how='left', left_on='Fecha_Ant',
                       right_on='Fecha')
data_input.drop(['Cierre_Actual', 'Fecha_Ant', 'Variacion', 'Fecha'],
                axis=1, inplace=True)
data_input = data_input[~data_input['Indicator'].isna()]
data_input['Fecha_Actual'] = data_input['Fecha_Actual'].apply(lambda x:
                                                              datetime.date(x))
data_input.set_index('Fecha_Actual', inplace=True)
data_input.sort_values(by='Fecha_Actual', inplace=True)

# =============================================================================
# GRÁFICOS
# =============================================================================
# PAIRPLOT
sns_pairplot = sns.pairplot(data_input, hue='Indicator',
                            # corner = True
                            )
# sns.set_style("white")
# sns.axes_style("white")
sns_pairplot.fig.suptitle("Relaciones bivariadas en cada clase", y=1.05, fontsize=30)
# Cuadrado
sns.set_context("paper", rc={"axes.titlesize": 5, "axes.labelsize": 20})
# Triangulo
# sns.set_context("paper", rc={"font.size":15,"axes.titlesize":15,"axes.labelsize":5})
# plt.legend(loc='upper left')
# plt.setp(.get_texts(), fontsize=20)
plt.xticks(rotation=70)
plt.show()
plt.savefig('images/features_pairplot.png')

# MODELADO
# Definir X, Y y el algoritmo
X = data_input.drop('Indicator', axis=1)
y = data_input.Indicator
X = X[1:]
y = y[1:]
clf = LinearSVC()
clf_2 = SVC(probability=True)  # Clasifica todos como 0
clf_3 = CalibratedClassifierCV(SVC())
# y_proba = clf.predict_proba(X_test)

# TRANSFORMACIÓN DE DATOS
X_trans = scale(X)
# VALIDACIÓN CRUZADA EN SERIE DE TIEMPO 
tscv = TimeSeriesSplit(n_splits=3)

# Ajustar modelos con varios hiperparámetros
parametros = {'penalty': ['l1', 'l2'],
              'C': [0.001, 0.1, 1, 10, 100],
              'class_weight': ['balanced', None],
              'max_iter': [1000, 100000]
              }
parametros_2 = {'C': [0.001, 0.1, 1, 10, 100],
                'kernel': ['linear', 'sigmoid'],
                'gamma': ['scale', 'auto'],
                'shrinking': [True, False],
                'class_weight': ['balanced', None],
                'max_iter': [1000, 100000]
                }
adj = RandomizedSearchCV(clf,
                         parametros,
                         cv=tscv,
                         random_state=0,
                         verbose=True,
                         scoring=['accuracy', 'precision'],
                         refit='accuracy')
adj_2 = RandomizedSearchCV(clf_2,
                           parametros_2,
                           cv=tscv,
                           random_state=0,
                           verbose=True,
                           scoring=['accuracy', 'precision'],
                           refit='accuracy')
# Ajustar modelos
clf_3.fit(X, y)
confusion_matrix(y, clf_3.predict(X))

adj.fit(X, y)
adj_2.fit(X_trans, y)
reporte = classification_report(y, adj.predict(X_trans), output_dict=True)
reporte_2 = classification_report(y, adj_2.predict(X_trans), output_dict=True)

"""
dump(adj, 'utils/modelo_direccion_precios.pkl')
gnb_from_pickle = pickle.loads(saved_model)
"""
# classifier = adj.best_estimator_
classifier = adj_2.best_estimator_
clf = adj_2.best_estimator_

# =============================================================================
# #Clasificador usando solo las variables financieras
# =============================================================================
# X_finan = X[['Cierre_Ant', 'Rolling_mean_10', 
#              'Rolling_mean_5', 'Rolling_mean_2',
#              'Rolling_std_10', 'Rolling_std_5', 'Rolling_std_2']]
# adj_2.fit(X_finan,y)
# clf = adj_2.best_estimator_
# confusion_matrix(y, clf.predict(X_finan))
# reporte = classification_report(y, clf.predict(X_finan), output_dict=True)

# n_splits = tscv.n_splits
# groups = np.hstack([[ii] * 83 for ii in range(3)])
# resultados = pd.DataFrame(columns = ['iteracion', 'train_size', 'test_size',
#                                      'acc_train', 'acc_test',
#                                      'prec_train', 'prec_test'])
# for ii, (tr, tt) in enumerate(tscv.split(X=X_finan, y=y, groups=groups)):
#     # Fill in indices with the training/test groups
#     # print(tr, tt)
#     X_train = X_finan.iloc[tr]
#     y_train = y[tr]
#     X_test = X_finan.iloc[tt]
#     y_test = y[tt]
#     cross_validacion = cross_validate(clf, 
#                                       X_train, 
#                                       y_train, 
#                                       cv = tscv, 
#                                       return_train_score=True,
#                                       scoring = ['accuracy', 'precision', 'recall'])
#     acc_train = cross_validacion['train_accuracy']
#     acc_test = cross_validacion['test_accuracy']
#     prec_train = cross_validacion['train_precision']
#     prec_test = cross_validacion['test_precision']
#     sens_train = cross_validacion['train_recall']
#     sens_test = cross_validacion['test_recall']
#     train_size = [len(X_train)]*3
#     test_size = [len(X_test)]*3
#     iteracion = ["C"+ str(ii)]*3
#     res_cv = pd.DataFrame({'iteracion': iteracion,
#                            'train_size': train_size,
#                            'test_size': test_size,
#                            'acc_train': acc_train,
#                            'acc_test': acc_test,
#                            'prec_train': prec_train,
#                            'prec_test': prec_test,
#                            'sens_train': sens_train,
#                            'sens_test': sens_test
#                            })
#     resultados = pd.concat([resultados, res_cv], axis=0, ignore_index=True)
# #CURVA ROC
# tprs = []
# aucs = []
# mean_fpr = np.linspace(0, 1, 100)

# fig, ax = plt.subplots(figsize=(5, 5))
# for i, (train, test) in enumerate(tscv.split(X_finan, y)):
#     clf.fit(X_finan.iloc[train], y.iloc[train])
#     viz = plot_roc_curve(clf, X_finan.iloc[test], y.iloc[test],
#                          name='ROC iteración {}'.format(i+1),
#                          alpha=0.7, lw=1, ax=ax)
#     #Interpolación lineal para multiplicar el número de puntos
#     interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
#     interp_tpr[0] = 0.0
#     tprs.append(interp_tpr)
#     aucs.append(viz.roc_auc)

# ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
#         label='Identidad', alpha=.8)

# mean_tpr = np.mean(tprs, axis=0)
# mean_tpr[-1] = 1.0
# mean_auc = auc(mean_fpr, mean_tpr)
# std_auc = np.std(aucs)
# ax.plot(mean_fpr, mean_tpr, color='b',
#         label=r'ROC promedio (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
#         lw=2, alpha=.8)
# std_tpr = np.std(tprs, axis=0)
# tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
# tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
# ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
#                 label=r'$\pm$ 1 std. dev.')

# ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
# ax.set_title("Curva ROC", fontsize=15)
# #ax.legend(loc=(1.05, 0.7), fontsize=8)
# ax.legend(loc='best', fontsize=8)
# ax.set_xlabel("Tasa de Falsos Positivos", fontsize=10)
# ax.set_ylabel("Tasa de Verdaderos Positivos", fontsize=10)
# plt.axis('equal')
# plt.show()
# plt.savefig('images/.png')

# CURVA DE APRENDIZAJE
fig, axes = plt.subplots(figsize=(20, 10))
plot_learning_curve(clf, 'SVM', X, y, axes=axes, ylim=(0.5, 1.01),
                    cv=tscv, n_jobs=-1)

# VALIDACIÓN CRUZADA EN SERIE DE TIEMPO
groups = np.hstack([[ii] * 25 for ii in range(10)])
cmap_cv = 'RdBu_r'
cmap_data = 'Blues_r'
# VALIDACIÓN CRUZADA EN SERIE DE TIEMPO
fig, ax = plt.subplots(figsize=(8, 5))
plot_cv_indices(tscv, X, y, groups, ax, 20, cmap_cv, cmap_data)
ax.legend([Patch(color='#2F79B5'), Patch(color='#C23639')],
          ['\nConjunto de \nentrenamiento', '\nConjunto de \nprueba\n'],
          loc=(1.01, 0.7))
plt.tight_layout()
fig.subplots_adjust(right=.7)
plt.show()
plt.savefig('images/ts_cv.png')

# DISTRIBUCIÓN DE VARIABLE OBJETIVO
balanceo = pd.Series(y).value_counts()
fig1, ax1 = plt.subplots()
ax1.pie(balanceo.values,
        labels=['Negativo', 'Positivo'],
        colors=['#79ADD2', '#F2B176'],
        autopct='%1.1f%%',
        shadow=False,
        startangle=90,
        wedgeprops=dict(width=0.6, edgecolor='w'))
ax1.set(aspect="equal", title='Balanceo del indicador de mercado')
ax1.axis('equal')  # Asegurar que el gráfico sea un círuclo
# ax1.legend(loc=(1,1))
plt.savefig('images/balanceo_indicador_mercado.png')

# NORMALIZACIÓN DE DATOS
# Variables en conteo
fig, axs = plt.subplots(10, 2, figsize=(8, 25), sharey=True)
axs[0, 0].hist(X['text_count'], bins=20, alpha=0.9, edgecolor='k')
axs[0, 0].set_title('Tweets (conteo)', fontsize=15)
axs[0, 0].tick_params(labelsize=15)
axs[1, 0].hist(X['Sentimiento_sum'], bins=20, alpha=0.9, edgecolor='k')
axs[1, 0].set_title('Sentimiento', fontsize=15)
axs[1, 0].tick_params(labelsize=15)
axs[2, 0].hist(X['Num_Palabras_sum'], bins=20, alpha=0.9, edgecolor='k')
axs[2, 0].set_title('Palabras (conteo)', fontsize=15)
axs[2, 0].tick_params(labelsize=15)
axs[3, 0].hist(X['Num_Caraceteres_sum'], bins=20, alpha=0.9, edgecolor='k')
axs[3, 0].set_title('Caracteres (conteo)', fontsize=15)
axs[3, 0].tick_params(labelsize=15)
axs[4, 0].hist(X['Num_Stopwords_sum'], bins=20, alpha=0.9, edgecolor='k')
axs[4, 0].set_title('Stopwords (conteo)', fontsize=15)
axs[4, 0].tick_params(labelsize=15)
axs[5, 0].hist(X['Num_Hashtags_sum'], bins=20, alpha=0.9, edgecolor='k')
axs[5, 0].set_title('Hashtags (conteo)', fontsize=15)
axs[5, 0].tick_params(labelsize=15)
axs[6, 0].hist(X['nbr_retweet_sum'], bins=20, alpha=0.9, edgecolor='k')
axs[6, 0].set_title('Retweets (conteo)', fontsize=15)
axs[6, 0].tick_params(labelsize=15)
axs[7, 0].hist(X['nbr_reply_sum'], bins=20, alpha=0.9, edgecolor='k')
axs[7, 0].set_title('Respuestas (conteo)', fontsize=15)
axs[7, 0].tick_params(labelsize=15)
axs[8, 0].hist(X['Num_Menciones_sum'], bins=20, alpha=0.9, edgecolor='k')
axs[8, 0].set_title('Menciones (conteo)', fontsize=15)
axs[8, 0].tick_params(labelsize=15)
axs[9, 0].hist(X['Num_Tickers_sum'], bins=20, alpha=0.9, edgecolor='k')
axs[9, 0].set_title('Cashtags (conteo)', fontsize=15)
axs[9, 0].tick_params(labelsize=15)
axs[0, 1].hist(X_trans[:, 24], bins=20, alpha=0.9, edgecolor='k')
axs[0, 1].set_title('Tweets normalizados (conteo)', fontsize=15)
axs[0, 1].tick_params(labelsize=15)
axs[1, 1].hist(X_trans[:, 11], bins=20, alpha=0.9, edgecolor='k')
axs[1, 1].set_title('Sentimiento normalizado (conteo)', fontsize=15)
axs[1, 1].tick_params(labelsize=15)
axs[2, 1].hist(X_trans[:, 12], bins=20, alpha=0.9, edgecolor='k')
axs[2, 1].set_title('Palabras normalizadas (conteo)', fontsize=15)
axs[2, 1].tick_params(labelsize=15)
axs[3, 1].hist(X_trans[:, 14], bins=20, alpha=0.9, edgecolor='k')
axs[3, 1].set_title('Caracteres normalizados (conteo)', fontsize=15)
axs[3, 1].tick_params(labelsize=15)
axs[4, 1].hist(X_trans[:, 16], bins=20, alpha=0.9, edgecolor='k')
axs[4, 1].set_title('Stopwords normalizadas (conteo)', fontsize=15)
axs[4, 1].tick_params(labelsize=15)
axs[5, 1].hist(X_trans[:, 18], bins=20, alpha=0.9, edgecolor='k')
axs[5, 1].set_title('Hashtags normalizados (conteo)', fontsize=15)
axs[5, 1].tick_params(labelsize=15)
axs[6, 1].hist(X_trans[:, 7], bins=20, alpha=0.9, edgecolor='k')
axs[6, 1].set_title('Retweets normalizados (conteo)', fontsize=15)
axs[6, 1].tick_params(labelsize=15)
axs[7, 1].hist(X_trans[:, 9], bins=20, alpha=0.9, edgecolor='k')
axs[7, 1].set_title('Respuestas normalizadas (conteo)', fontsize=15)
axs[7, 1].tick_params(labelsize=15)
axs[8, 1].hist(X_trans[:, 20], bins=20, alpha=0.9, edgecolor='k')
axs[8, 1].set_title('Menciones normalizadas (conteo)', fontsize=15)
axs[8, 1].tick_params(labelsize=15)
axs[9, 1].hist(X_trans[:, 22], bins=20, alpha=0.9, edgecolor='k')
axs[9, 1].set_title('Cashtags normalizados (conteo)', fontsize=15)
axs[9, 1].tick_params(labelsize=15)
axs[1, 1].get_shared_x_axes().join(axs[0, 1], axs[1, 1], axs[2, 1], axs[3, 1],
                                   axs[4, 1], axs[5, 1], axs[6, 1], axs[7, 1],
                                   axs[8, 1], axs[9, 1])
axs[1, 1].autoscale()
axs[2, 1].autoscale()
axs[3, 1].autoscale()
axs[4, 1].autoscale()
axs[5, 1].autoscale()
axs[6, 1].autoscale()
axs[7, 1].autoscale()
axs[8, 1].autoscale()
axs[9, 1].autoscale()
axs[9, 1].tick_params(labelsize=15)
fig.suptitle('Normalización de variables de conteo', y=1.02, x=0.5, fontsize=20)
plt.tight_layout()
plt.savefig('images/norm_variables_01.png')

# Variables en Promedio
fig, axs = plt.subplots(8, 2, figsize=(8, 25), sharey=True)
axs[0, 0].hist(X['Num_Palabras_mean'], bins=20, alpha=0.9, edgecolor='k')
axs[0, 0].set_title('Palabras (promedio)', fontsize=15)
axs[0, 0].tick_params(labelsize=15)
axs[1, 0].hist(X['Num_Caraceteres_mean'], bins=20, alpha=0.9, edgecolor='k')
axs[1, 0].set_title('Caracteres (promedio)', fontsize=15)
axs[1, 0].tick_params(labelsize=15)
axs[2, 0].hist(X['Num_Stopwords_mean'], bins=20, alpha=0.9, edgecolor='k')
axs[2, 0].set_title('Stopwords (promedio)', fontsize=15)
axs[2, 0].tick_params(labelsize=15)
axs[3, 0].hist(X['Num_Hashtags_mean'], bins=20, alpha=0.9, edgecolor='k')
axs[3, 0].set_title('Hashtags (promedio)', fontsize=15)
axs[3, 0].tick_params(labelsize=15)
axs[4, 0].hist(X['nbr_retweet_mean'], bins=20, alpha=0.9, edgecolor='k')
axs[4, 0].set_title('Retweets (promedio)', fontsize=15)
axs[4, 0].tick_params(labelsize=15)
axs[5, 0].hist(X['nbr_reply_mean'], bins=20, alpha=0.9, edgecolor='k')
axs[5, 0].set_title('Respuestas (promedio)', fontsize=15)
axs[5, 0].tick_params(labelsize=15)
axs[6, 0].hist(X['Num_Menciones_mean'], bins=20, alpha=0.9, edgecolor='k')
axs[6, 0].set_title('Menciones (promedio)', fontsize=15)
axs[6, 0].tick_params(labelsize=15)
axs[7, 0].hist(X['Num_Tickers_mean'], bins=20, alpha=0.9, edgecolor='k')
axs[7, 0].set_title('Cashtags (promedio)', fontsize=15)
axs[7, 0].tick_params(labelsize=15)
axs[0, 1].hist(X_trans[:, 13], bins=20, alpha=0.9, edgecolor='k')
axs[0, 1].set_title('Palabras normalizadas (promedio)', fontsize=15)
axs[0, 1].tick_params(labelsize=15)
axs[1, 1].hist(X_trans[:, 15], bins=20, alpha=0.9, edgecolor='k')
axs[1, 1].set_title('Caracteres normalizados (promedio)', fontsize=15)
axs[1, 1].tick_params(labelsize=15)
axs[2, 1].hist(X_trans[:, 17], bins=20, alpha=0.9, edgecolor='k')
axs[2, 1].set_title('Stopwords normalizadas (promedio)', fontsize=15)
axs[2, 1].tick_params(labelsize=15)
axs[3, 1].hist(X_trans[:, 19], bins=20, alpha=0.9, edgecolor='k')
axs[3, 1].set_title('Hashtags normalizados (promedio)', fontsize=15)
axs[3, 1].tick_params(labelsize=15)
axs[4, 1].hist(X_trans[:, 8], bins=20, alpha=0.9, edgecolor='k')
axs[4, 1].set_title('Retweets normalizados (promedio)', fontsize=15)
axs[4, 1].tick_params(labelsize=15)
axs[5, 1].hist(X_trans[:, 10], bins=20, alpha=0.9, edgecolor='k')
axs[5, 1].set_title('Respuestas normalizadas (promedio)', fontsize=15)
axs[5, 1].tick_params(labelsize=15)
axs[6, 1].hist(X_trans[:, 21], bins=20, alpha=0.9, edgecolor='k')
axs[6, 1].set_title('Menciones normalizadas (promedio)', fontsize=15)
axs[6, 1].tick_params(labelsize=15)
axs[7, 1].hist(X_trans[:, 23], bins=20, alpha=0.9, edgecolor='k')
axs[7, 1].set_title('Cashtags normalizados (promedio)', fontsize=15)
axs[7, 1].tick_params(labelsize=15)
axs[1, 1].get_shared_x_axes().join(axs[0, 1], axs[1, 1], axs[2, 1], axs[3, 1],
                                   axs[4, 1], axs[5, 1], axs[6, 1], axs[7, 1])
axs[1, 1].autoscale()
axs[2, 1].autoscale()
axs[3, 1].autoscale()
axs[4, 1].autoscale()
axs[5, 1].autoscale()
axs[6, 1].autoscale()
axs[7, 1].autoscale()
axs[7, 1].tick_params(labelsize=15)
fig.suptitle('Normalización de variables promediadas', y=1.02, x=0.5, fontsize=20)
plt.tight_layout()
plt.savefig('images/norm_variables_02.png')

# Variables de Precio
fig, axs = plt.subplots(7, 2, figsize=(8, 25), sharey=True)
axs[0, 0].hist(X['Cierre_Ant'], bins=20, alpha=0.9, edgecolor='k')
axs[0, 0].set_title('Precio del día anterior', fontsize=15)
axs[0, 0].tick_params(labelsize=15)
axs[1, 0].hist(X['Rolling_mean_10'], bins=20, alpha=0.9, edgecolor='k')
axs[1, 0].set_title('Precio prom. de 10 días', fontsize=15)
axs[1, 0].tick_params(labelsize=15)
axs[2, 0].hist(X['Rolling_mean_5'], bins=20, alpha=0.9, edgecolor='k')
axs[2, 0].set_title('Precio prom. de 5 días', fontsize=15)
axs[2, 0].tick_params(labelsize=15)
axs[3, 0].hist(X['Rolling_mean_2'], bins=20, alpha=0.9, edgecolor='k')
axs[3, 0].set_title('Precio prom. de 2 días', fontsize=15)
axs[3, 0].tick_params(labelsize=15)
axs[4, 0].hist(X['Rolling_std_10'], bins=20, alpha=0.9, edgecolor='k')
axs[4, 0].set_title('Desv. est. de 10 días', fontsize=15)
axs[4, 0].tick_params(labelsize=15)
axs[5, 0].hist(X['Rolling_std_5'], bins=20, alpha=0.9, edgecolor='k')
axs[5, 0].set_title('Desv. est. de 5 días', fontsize=15)
axs[5, 0].tick_params(labelsize=15)
axs[6, 0].hist(X['Rolling_std_2'], bins=20, alpha=0.9, edgecolor='k')
axs[6, 0].set_title('Desv. est. de 2 días', fontsize=15)
axs[6, 0].tick_params(labelsize=15)
axs[0, 1].hist(X_trans[:, 0], bins=20, alpha=0.9, edgecolor='k')
axs[0, 1].set_title('Precio normalizado (día anterior)', fontsize=15)
axs[0, 1].tick_params(labelsize=15)
axs[1, 1].hist(X_trans[:, 1], bins=20, alpha=0.9, edgecolor='k')
axs[1, 1].set_title('Precio prom. normalizado (10 días)', fontsize=15)
axs[1, 1].tick_params(labelsize=15)
axs[2, 1].hist(X_trans[:, 2], bins=20, alpha=0.9, edgecolor='k')
axs[2, 1].set_title('Precio prom. normalizado (5 días)', fontsize=15)
axs[2, 1].tick_params(labelsize=15)
axs[3, 1].hist(X_trans[:, 3], bins=20, alpha=0.9, edgecolor='k')
axs[3, 1].set_title('Precio prom. normalizado (2 días)', fontsize=15)
axs[3, 1].tick_params(labelsize=15)
axs[4, 1].hist(X_trans[:, 4], bins=20, alpha=0.9, edgecolor='k')
axs[4, 1].set_title('Desv. est. normalizada (10 días)', fontsize=15)
axs[4, 1].tick_params(labelsize=15)
axs[5, 1].hist(X_trans[:, 5], bins=20, alpha=0.9, edgecolor='k')
axs[5, 1].set_title('Desv. est. normalizada (5 días)', fontsize=15)
axs[5, 1].tick_params(labelsize=15)
axs[6, 1].hist(X_trans[:, 6], bins=20, alpha=0.9, edgecolor='k')
axs[6, 1].set_title('Desv. est. normalizada (2 días)', fontsize=15)
axs[6, 1].tick_params(labelsize=15)
axs[1, 1].get_shared_x_axes().join(axs[0, 1], axs[1, 1], axs[2, 1], axs[3, 1],
                                   axs[4, 1], axs[5, 1], axs[6, 1])
axs[1, 1].autoscale()
axs[2, 1].autoscale()
axs[3, 1].autoscale()
axs[4, 1].autoscale()
axs[5, 1].autoscale()
axs[6, 1].autoscale()
axs[6, 1].tick_params(labelsize=15)
fig.suptitle('Normalización de variables de precio', y=1.02, x=0.5, fontsize=20)
plt.tight_layout()
plt.savefig('images/norm_variables_03.png')

# LEARNING CURVE
n_splits = tscv.n_splits
groups = np.hstack([[ii] * 83 for ii in range(3)])
resultados = pd.DataFrame(columns=['iteracion', 'train_size', 'test_size',
                                   'acc_train', 'acc_test',
                                   'prec_train', 'prec_test'])
for ii, (tr, tt) in enumerate(tscv.split(X=X_trans, y=y, groups=groups)):
    # Fill in indices with the training/test groups
    # print(tr, tt)
    X_train = X_trans[tr]
    y_train = y[tr]
    X_test = X_trans[tt]
    y_test = y[tt]
    cross_validacion = cross_validate(clf,
                                      X_train,
                                      y_train,
                                      cv=tscv,
                                      return_train_score=True,
                                      scoring=['accuracy', 'precision', 'recall'])
    acc_train = cross_validacion['train_accuracy']
    acc_test = cross_validacion['test_accuracy']
    prec_train = cross_validacion['train_precision']
    prec_test = cross_validacion['test_precision']
    sens_train = cross_validacion['train_recall']
    sens_test = cross_validacion['test_recall']
    train_size = [len(X_train)] * 3
    test_size = [len(X_test)] * 3
    iteracion = ["C" + str(ii)] * 3
    res_cv = pd.DataFrame({'iteracion': iteracion,
                           'train_size': train_size,
                           'test_size': test_size,
                           'acc_train': acc_train,
                           'acc_test': acc_test,
                           'prec_train': prec_train,
                           'prec_test': prec_test,
                           'sens_train': sens_train,
                           'sens_test': sens_test
                           })
    resultados = pd.concat([resultados, res_cv], axis=0, ignore_index=True)
train_acc_mean = resultados.groupby('iteracion')['acc_train'].mean()
train_acc_std = resultados.groupby('iteracion')['acc_train'].std()
test_acc_mean = resultados.groupby('iteracion')['acc_test'].mean()
test_acc_std = resultados.groupby('iteracion')['acc_test'].std()
train_prec_mean = resultados.groupby('iteracion')['prec_train'].mean()
train_prec_std = resultados.groupby('iteracion')['prec_train'].std()
test_prec_mean = resultados.groupby('iteracion')['prec_test'].mean()
test_prec_std = resultados.groupby('iteracion')['prec_test'].std()
train_sens_mean = resultados.groupby('iteracion')['sens_train'].mean()
train_sens_std = resultados.groupby('iteracion')['sens_train'].std()
test_sens_mean = resultados.groupby('iteracion')['sens_test'].mean()
test_sens_std = resultados.groupby('iteracion')['sens_test'].std()

# Exactitud
_, axes = plt.subplots(figsize=(8, 6))
axes.set_title('Curva de Aprendizaje', fontsize=20)
axes.set_xlabel("Observaciones de entrenamiento", fontsize=15)
axes.set_ylabel("Exactitud", fontsize=15)
axes.fill_between(list(resultados['train_size'].unique()),
                  train_acc_mean - train_acc_std,
                  train_acc_mean + train_acc_std,
                  alpha=0.1,
                  color='#2F78B4')
axes.fill_between(list(resultados['train_size'].unique()),
                  test_acc_mean - test_acc_std,
                  test_acc_mean + test_acc_std,
                  alpha=0.1,
                  color='#C13639')
axes.plot(list(resultados['train_size'].unique()),
          list(train_acc_mean), 'o-', color="#2F78B4",
          label=" \n Conjunto de \n entrenamiento \n",
          lw=2)
axes.plot(list(resultados['train_size'].unique()),
          list(test_acc_mean), 'o-', color="#C13639",
          label="Conjunto de \n prueba",
          lw=2)
axes.legend(loc=(1.01, 0.70), fontsize=15)
plt.show()
plt.savefig('images/curva_aprendizaje_exactitud.png')

# Precision
_, axes = plt.subplots(figsize=(8, 6))
axes.set_title('Curva de Aprendizaje', fontsize=15)
axes.set_xlabel("Observaciones de entrenamiento", fontsize=10)
axes.set_ylabel("Precisión", fontsize=10)
axes.fill_between(list(resultados['train_size'].unique()),
                  train_prec_mean - train_prec_std,
                  train_prec_mean + train_prec_std,
                  alpha=0.1,
                  color='#2F78B4')
axes.fill_between(list(resultados['train_size'].unique()),
                  test_prec_mean - test_prec_std,
                  test_prec_mean + test_prec_std,
                  alpha=0.1,
                  color='#C13639')
axes.plot(list(resultados['train_size'].unique()),
          list(train_prec_mean), 'o-', color="#2F78B4",
          label=" \n Conjunto de \n entrenamiento \n",
          lw=2)
axes.plot(list(resultados['train_size'].unique()),
          list(test_prec_mean), 'o-', color="#C13639",
          label="Conjunto de \n prueba",
          lw=2)
axes.legend(loc=(1.01, 0.70), fontsize=10)
plt.show()
plt.savefig('images/curva_aprendizaje_precision.png')

# =============================================================================
# Curvas de aprendizaje reunidas: Exactitud, Precisión y Sensibilidad
# =============================================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
fig.suptitle('Curva de Aprendizaje', fontsize=12)
# Exactitud
ax1.set_xlabel("Observaciones de entrenamiento", fontsize=8)
ax1.set_ylabel("Exactitud", fontsize=10)
ax1.fill_between(list(resultados['train_size'].unique()),
                 train_acc_mean - train_acc_std,
                 train_acc_mean + train_acc_std,
                 alpha=0.1,
                 color='#2F78B4')
ax1.fill_between(list(resultados['train_size'].unique()),
                 test_acc_mean - test_acc_std,
                 test_acc_mean + test_acc_std,
                 alpha=0.1,
                 color='#C13639')
ax1.plot(list(resultados['train_size'].unique()),
         list(train_acc_mean), 'o-', color="#2F78B4",
         label=" \n Conjunto de \n entrenamiento \n",
         lw=2)
ax1.plot(list(resultados['train_size'].unique()),
         list(test_acc_mean), 'o-', color="#C13639",
         label="Conjunto de \n prueba",
         lw=2)
# Precisión
ax2.set_xlabel("Observaciones de entrenamiento", fontsize=8)
ax2.set_ylabel("Precisión", fontsize=10)
ax2.fill_between(list(resultados['train_size'].unique()),
                 train_prec_mean - train_prec_std,
                 train_prec_mean + train_prec_std,
                 alpha=0.1,
                 color='#2F78B4')
ax2.fill_between(list(resultados['train_size'].unique()),
                 test_prec_mean - test_prec_std,
                 test_prec_mean + test_prec_std,
                 alpha=0.1,
                 color='#C13639')
ax2.plot(list(resultados['train_size'].unique()),
         list(train_prec_mean), 'o-', color="#2F78B4",
         lw=2)
ax2.plot(list(resultados['train_size'].unique()),
         list(test_prec_mean), 'o-', color="#C13639",
         lw=2)
# Sensibilidad
ax3.set_xlabel("Observaciones de entrenamiento", fontsize=8)
ax3.set_ylabel("Sensibilidad", fontsize=10)
ax3.fill_between(list(resultados['train_size'].unique()),
                 train_sens_mean - train_sens_std,
                 train_sens_mean + train_sens_std,
                 alpha=0.1,
                 color='#2F78B4')
ax3.fill_between(list(resultados['train_size'].unique()),
                 test_sens_mean - test_sens_std,
                 test_sens_mean + test_sens_std,
                 alpha=0.1,
                 color='#C13639')
ax3.plot(list(resultados['train_size'].unique()),
         list(train_sens_mean), 'o-', color="#2F78B4",
         lw=2)
ax3.plot(list(resultados['train_size'].unique()),
         list(test_sens_mean), 'o-', color="#C13639",
         lw=2)
fig.legend(fontsize=7, loc='center right')
plt.savefig('images/curva_aprendizaje.png')

# =============================================================================
# CURVA DE VALIDACION
# =============================================================================

# Parámetro de regularización
param_range = np.linspace(0.0001, 100, 50)

# ACCURACY
train_scores, test_scores = validation_curve(
    clf, X_trans, y, param_name="C", param_range=param_range,
    scoring="accuracy", n_jobs=1)
train_accu_mean = np.mean(train_scores, axis=1)
train_accu_std = np.std(train_scores, axis=1)
test_accu_mean = np.mean(test_scores, axis=1)
test_accu_std = np.std(test_scores, axis=1)
# PRECISION
train_scores, test_scores = validation_curve(
    clf, X_trans, y, param_name="C", param_range=param_range,
    scoring="precision", n_jobs=1)
train_prec_mean = np.mean(train_scores, axis=1)
train_prec_std = np.std(train_scores, axis=1)
test_prec_mean = np.mean(test_scores, axis=1)
test_prec_std = np.std(test_scores, axis=1)
# SENSIBILIDAD
train_scores, test_scores = validation_curve(
    clf, X_trans, y, param_name="C", param_range=param_range,
    scoring="recall", n_jobs=1)
train_sens_mean = np.mean(train_scores, axis=1)
train_sens_std = np.std(train_scores, axis=1)
test_sens_mean = np.mean(test_scores, axis=1)
test_sens_std = np.std(test_scores, axis=1)

_, axes = plt.subplots(figsize=(10, 8))
axes.set_title("Curva de validación: Regularización", fontsize=20)
axes.set_xlabel("C", fontsize=15)
axes.set_ylabel("Exactitud", fontsize=15)
# axes.set_ylim(0.0, 1.1)
axes.semilogx(param_range, train_accu_mean,
              label="Conjunto de \n entrenamiento",
              color="#2F78B4")
axes.fill_between(param_range, train_accu_mean - train_accu_std,
                  train_accu_mean + train_accu_std, alpha=0.1,
                  color="#2F78B4", lw=2)
axes.semilogx(param_range, test_accu_mean,
              label="Validación cruzada",
              color="#C13639", lw=2)
axes.fill_between(param_range, test_accu_mean - test_accu_std,
                  test_accu_mean + test_accu_std, alpha=0.1,
                  color="#C13639")
axes.legend(loc=(1.01, 0.85), fontsize=15)
# plt.show()
plt.savefig('images/curva_validacion_c_acc.png')

# Plot de precision
_, axes = plt.subplots(figsize=(10, 8))
axes.set_title("Curva de validación: Regularización", fontsize=20)
axes.set_xlabel("C", fontsize=15)
axes.set_ylabel("Precisión", fontsize=15)
# axes.set_ylim(0.0, 1.1)
axes.semilogx(param_range, train_prec_mean,
              label="Conjunto de \n entrenamiento",
              color="#2F78B4")
axes.fill_between(param_range, train_prec_mean - train_prec_std,
                  train_prec_mean + train_prec_std, alpha=0.1,
                  color="#2F78B4", lw=2)
axes.semilogx(param_range, test_prec_mean,
              label="Validación cruzada",
              color="#C13639", lw=2)
axes.fill_between(param_range, test_prec_mean - test_prec_std,
                  test_prec_mean + test_prec_std, alpha=0.1,
                  color="#C13639")
axes.legend(loc=(1.01, 0.85), fontsize=15)
# plt.show()
plt.savefig('images/curva_validacion_c_prec.png')

# Curvas de validación reunidas: Exactitud, Precisión y Sensibilidad
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
fig.suptitle('Curva de validación: Regularización', fontsize=12)
ax1.set_xlabel("Regularización", fontsize=8)
ax1.set_ylabel("Exactitud", fontsize=10)
ax1.semilogx(param_range, train_accu_mean,
             label="Conjunto de \n entrenamiento",
             color="#2F78B4")
ax1.fill_between(param_range, train_accu_mean - train_accu_std,
                 train_accu_mean + train_accu_std, alpha=0.1,
                 color="#2F78B4", lw=2)
ax1.semilogx(param_range, test_accu_mean,
             label="Validación \n cruzada",
             color="#C13639", lw=2)
ax1.fill_between(param_range, test_accu_mean - test_accu_std,
                 test_accu_mean + test_accu_std, alpha=0.1,
                 color="#C13639")
# Precisión
ax2.set_xlabel("Regularización", fontsize=8)
ax2.set_ylabel("Precisión", fontsize=10)
ax2.semilogx(param_range, train_prec_mean,
             color="#2F78B4")
ax2.fill_between(param_range, train_prec_mean - train_prec_std,
                 train_prec_mean + train_prec_std, alpha=0.1,
                 color="#2F78B4", lw=2)
ax2.semilogx(param_range, test_prec_mean,
             color="#C13639", lw=2)
ax2.fill_between(param_range, test_prec_mean - test_prec_std,
                 test_prec_mean + test_prec_std, alpha=0.1,
                 color="#C13639")
# Sensibilidad
ax3.set_xlabel("Regularización", fontsize=8)
ax3.set_ylabel("Sensibilidad", fontsize=10)
ax3.semilogx(param_range, train_sens_mean,
             color="#2F78B4")
ax3.fill_between(param_range, train_sens_mean - train_sens_std,
                 train_sens_mean + train_sens_std, alpha=0.1,
                 color="#2F78B4", lw=2)
ax3.semilogx(param_range, test_sens_mean,
             color="#C13639", lw=2)
ax3.fill_between(param_range, test_sens_mean - test_sens_std,
                 test_sens_mean + test_sens_std, alpha=0.1,
                 color="#C13639")
fig.legend(fontsize=7, loc='center right')
plt.savefig('images/curva_validacion_c.png')

# CURVA ROC
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
fig, ax = plt.subplots(figsize=(5, 5))
for i, (train, test) in enumerate(tscv.split(X_trans, y)):
    classifier.fit(X_trans[train], y.iloc[train])
    viz = plot_roc_curve(classifier, X_trans[test], y.iloc[test],
                         name='ROC iteración {}'.format(i + 1),
                         alpha=0.7, lw=1, ax=ax)
    # Interpolación lineal para multiplicar el número de puntos
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)
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
                label=r'$\pm$ 1 std. dev.')
ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
ax.set_title("Curva ROC", fontsize=15)
# ax.legend(loc=(1.05, 0.7), fontsize=8)
ax.legend(loc='best', fontsize=8)
ax.set_xlabel("Tasa de Falsos Positivos", fontsize=10)
ax.set_ylabel("Tasa de Verdaderos Positivos", fontsize=10)
plt.axis('equal')
# plt.show()
plt.savefig('images/roc_oportunidades.png')

# CURVA PRECISIÓN-RECALL
draw_cv_pr_curve(classifier, tscv, X_trans, y, title='Curva PR')

# =============================================================================
# CONFUSION MATRIX: CAMBIO EN THRESHOLD
# =============================================================================
des_func = classifier.fit(X_trans, y).decision_function(X_trans)
cm_1 = confusion_matrix(y, classifier.predict(X_trans))
cm_2 = confusion_matrix(y, np.where(des_func > 0.5, 1, 0))
cm_3 = confusion_matrix(y, np.where(des_func > -0.5, 1, 0))
for cm, df in zip([cm_1, cm_2, cm_3], ['0', '0.5', '-0.5']):
    plt.clf()
    ax = sns.heatmap(cm, annot=True, cmap='Blues_r',
                     fmt='g', cbar=False, linewidths=2,
                     annot_kws={"size": 18}, square=True)
    plt.title('Matriz de Confusión \nFunción de Decisión = ' + df, fontsize=15)
    plt.xlabel('Predicción', fontsize=10)
    plt.ylabel('Real', fontsize=10)
    plt.savefig(r'images/conf_matriz_' + df + '.png')
    # plt.show()
f1_score(y, classifier.predict(X_trans))
f1_score(y, np.where(des_func > 0.5, 1, 0))
f1_score(y, np.where(des_func > -0.5, 1, 0))
