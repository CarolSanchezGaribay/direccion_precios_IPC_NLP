import os
import pandas as pd
import json
import seaborn as sns
import nltk
import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import re
from string import punctuation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import warnings

warnings.filterwarnings("ignore")
# nltk.download("stopwords")
print("Directorio actual: " + os.getcwd())

# =============================================================================
# INPUTS
# =============================================================================
CLAS_MANUAL_2 = r'data/tweets_data_trained_2.xlsx'
CLAS_MANUAL_3 = r'data/tweets_data_trained_3.xlsx'
SUP_TUITS = r'data/supervised_tweets.csv'
LEMAS = 'utils/lemma-es.csv'

# =============================================================================
# FUNCIONES
# =============================================================================


def carga_tweets(ruta):
    tipo = []
    tweets_cargados = []
    archivos = os.listdir(ruta)
    for arch in archivos:
        print(arch)
        if arch != '.DS_Store':
            ruta_arch = ruta + arch + '/'
            docs = os.listdir(ruta_arch)
            for doc in docs:
                if doc != 'Users' and not doc.startswith('.'):
                    with open(ruta_arch + doc, "r", encoding='utf-8',
                              errors='ignore') as read_file:
                        tweet = json.loads(read_file.read())
                        tweets_cargados.append(tweet)
                        tipo.append(arch)
    return tweets_cargados, tipo


# NORMALIZACION                                                                  


def elimina_acentos(palabra):
    # Corregir encoding
    x_encoding = (("√°", "á"), ("√©", "é"), ("√≠", "í"),
                  ("√≥", "ó"), ("√∫", "ú"), ("√±", "ñ"))
    for a, b in x_encoding:
        palabra = palabra.replace(a, b).replace(a.upper(), b.upper())
    # Eliminar acentos
    acentos = (("á", "a"), ("é", "e"), ("í", "i"), ("ó", "o"), ("ú", "u"))
    for a, b in acentos:
        palabra = palabra.replace(a, b).replace(a.upper(), b.upper())
    return palabra


def normalizar(texto):
    texto = str(texto)
    texto = texto.lower()
    texto = elimina_acentos(texto)
    texto = re.sub(r'\d+', '', texto)  # Eliminar números
    texto = " ".join(x for x in texto.split() if '/' not in x)
    texto = re.sub(r'[¿?¡!]', '', texto)  # Eliminar signos de puntuación
    texto = re.sub(r'\b[a-z]\b', '', texto)
    texto = texto.replace('fb', 'facebook')
    texto = texto.replace('hrs', 'horas')
    texto = texto.replace('mx', 'mexico')
    texto = texto.replace('info', 'informacion')
    texto = texto.replace('rajoy', 'presidente')
    texto = texto.replace('psoe', 'partido politico')
    texto = " ".join(x for x in texto.split() if not x.startswith('www.'))
    texto = " ".join(x for x in texto.split() if not x.startswith('http'))
    texto = " ".join(x for x in texto.split() if not x.startswith('pic.twitter'))
    texto = " ".join(x for x in texto.split() if not len(x) > 14)
    texto = re.sub('[%s]' % re.escape(punctuation), '', texto)
    texto = re.sub(r'(?=[^aeiou]{4})[a-z]{4,}', '', texto)
    texto = re.sub(r'\s+', ' ', texto)
    texto = re.sub(r'“', '', texto)
    texto = re.sub(r'”', '', texto)
    texto = " ".join(x for x in texto.split() if x not in non_words)
    texto = lematizacion(texto)
    texto = texto.strip()
    return texto


def lematizacion(texto):
    texto = " ".join(dict_lemas[x] if x in dict_lemas.keys()
                     else x for x in texto.split())
    return texto


# STOPWORDS


def stop_words_esp():
    spanish_stopwords = stopwords.words('spanish')
    spanish_stopwords.append('mas')
    non_words = list(punctuation)
    non_words.extend(['¿', '¡', '“', '”', '€', '▸', '«', '»',
                      '°', '⁩', '’', '‘',
                      'jaja', 'jajaja', 'jeje', 'pp', 'rt', 'xd',
                      'pq', 'tb', 'xq', 'zp', 'pd', 'sp', 'dc', 'mm',
                      'pb', 'ls', 'pa',
                      '…', 'mr', 'qtf', 'nq', 'in', 'tras',
                      'sino', 'cada', 'via',
                      'asi', 'aqui', 'ademas', 'san', 'traves', 'aun',
                      'ciento', 'mil', 'millones',
                      'dos', 'tres', 'cuatro', 'cinco', 'seis',
                      'siete', 'ocho', 'nueve'
                                       'fm', 'am', 'video', 'excelsiortv',
                      'todaslasvoces', 'clubdetraders', 'suscribete',
                      'wradiomexico', 'laveoonolaveo', 'envivo',
                      'porsinoloviste', 'sinanestesia', 'wradiocommx',
                      'radioformula', 'duendepregunton', 'realdonaldtrump',
                      'datacoparmex', 'jesusmartinmx', 'agronoticiasmx',
                      'mileniotv', 'felizmiercoles', 'invierteenbolsa',
                      'actuacoparmex', 'elunimxhcudtmf'])
    non_words.extend(map(str, range(10)))
    non_words.extend(spanish_stopwords)
    non_words.remove('ni')
    non_words.remove('no')
    non_words.remove('sin')
    non_words.remove('mas')
    non_words.remove('sí')
    non_words.remove('contra')
    for palabra in non_words:
        if "á" in palabra or "é" in palabra \
                or "í" in palabra or "ó" in palabra or "ú" in palabra:
            palabra_sin_acento = elimina_acentos(palabra)
            non_words.append(palabra_sin_acento)
    return non_words


# LEMAS

lemas = pd.read_csv(LEMAS, header=None)
non_words = stop_words_esp()
lemas.columns = ["LEMA", "PALABRA"]
lemas['PALABRA'] = lemas['PALABRA'].apply(lambda x: elimina_acentos(x))
lemas['LEMA'] = lemas['LEMA'].apply(lambda x: elimina_acentos(x))
dict_lemas = pd.Series(lemas['LEMA'].values, index=lemas['PALABRA']).to_dict()


# TOKENIZACION


def tokenize(text):
    tokens = word_tokenize(text)
    text = ' '.join([c for c in tokens if c not in non_words])
    tokens = word_tokenize(text)
    return tokens


# INGENIERIA


def prom_palabras(tuit):
    words = tuit.split()
    return sum(len(word) for word in words) / len(words)


def num_stopwords(tuit):
    stops = []
    spanish_stopwords = stopwords.words('spanish')
    tokens = word_tokenize(tuit)
    for palabra in tokens:
        if palabra in spanish_stopwords:
            stops.append(palabra)
    return len(stops)


# Diccionario de días
dict_weekday = {'Monday': 'Lunes',
                'Tuesday': 'Martes',
                'Wednesday': 'Miércoles',
                'Thursday': 'Jueves',
                'Friday': 'Viernes',
                'Saturday': 'Sábado',
                'Sunday': 'Domingo'
                }

# =============================================================================

if __name__ == "__main__":
    print("Leyendo información")
    data = pd.read_csv(SUP_TUITS, encoding='utf-8')
    data['datetime'] = pd.to_datetime(data['datetime'], errors='coerce')
    data = data[(data['datetime']
                 >= datetime.datetime(2018, 10, 1)) &
                (data['datetime']
                 <= datetime.datetime(2019, 9, 30))].sort_values(by=['datetime'])
    data['date'] = data['datetime'].dt.date
    data['time'] = data['datetime'].dt.time
    data['time_2'] = data['time'] \
        .apply(lambda x: datetime.time(x.hour, x.minute, 0, 0))
    data['time_2'] = data['time_2'] \
        .apply(lambda x: datetime.time(x.hour, int(round(x.minute, -1)) % 60, 0, 0))
    data['time_3'] = data['datetime'].dt.round('H').dt.hour
    data['weekday'] = data['date'].apply(lambda x: x.strftime('%A'))
    data['weekday'] = data['weekday'].apply(lambda x: dict_weekday[x])
    data['text'] = data['text'].apply(lambda x: x.replace('# ', '#'))
    data['text'] = data['text'].apply(lambda x: x.replace('$ ', '$'))
    data['text'] = data['text'].apply(lambda x: x.replace('@ ', '@'))
    data['Num_Palabras'] = data['text'].apply(lambda x: len(str(x).split(" ")))
    data['Num_Caraceteres'] = data['text'].str.len()
    data['Num_Stopwords'] = data['text'].apply(lambda x: num_stopwords(x))
    data['Largo_promedio'] = data['text'].apply(lambda x: prom_palabras(x))
    data['Num_Hashtags'] = data['text'].apply(lambda x:
                                              len([x for x in str(x).split()
                                                   if x.startswith('#')]))
    data['Num_Menciones'] = data['text'].apply(lambda x:
                                               len([x for x in str(x).split()
                                                    if x.startswith('@')]))
    data['Num_Tickers'] = data['text'].apply(lambda x:
                                             len([x for x in str(x).split()
                                                  if x.startswith('$')]))
    data.to_csv('data/supervised_tweets_all_variables.csv',
                sep=',',
                index=False,
                encoding='utf-8')
    data['cuenta'] = 1
    # TOKENS' COUNT
    textos = data['text'].drop_duplicates()
    # textos = textos.apply(lambda x: normalizar(x))
    tokens = textos.apply(lambda x: nltk.word_tokenize(str(x), "spanish"))
    list_tokens = [palabra for token in tokens for palabra in token]
    tokens_df = pd.DataFrame(list_tokens, columns=['token'])
    tokens_df['Count'] = 1
    tokens_df = tokens_df \
        .groupby('token', as_index=False) \
        .count().sort_values(by='Count', ascending=False) \
        .reset_index(drop=True)
    tokens_df = tokens_df[~tokens_df['token'].isin([':', '//', ',', '#', 'http', '.', '@'])] \
        .reset_index(drop=True)
    tokens_no_stopwords = textos.apply(lambda x: normalizar(x))
    tokens_no_stopwordss = tokens_no_stopwords.apply(lambda x:
                                                     tokenize(str(x)))
    list_tokens_no_sw = [palabra for token in tokens_no_stopwordss
                         for palabra in token]
    tokens_no_sw_df = pd.DataFrame(list_tokens_no_sw, columns=['token'])
    tokens_no_sw_df['Count'] = 1
    tokens_no_sw_df = tokens_no_sw_df \
        .groupby('token', as_index=False) \
        .count().sort_values(by='Count', ascending=False) \
        .reset_index(drop=True)
    textos = textos.apply(lambda x: normalizar(x))
    # Coeficiente de asimetría
    data.skew()

    # =========================================================================
    # GRÁFICOS
    # ========================================================================
    print("Generando gráficos")
    # Comparativo de palabras más frecuentes
    print("    *Palabras muy frecuentes")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle('Palabras más frecuentes', fontsize=15, y=1.05)
    ax1.barh(tokens_df['token'][0:10], tokens_df['Count'][0:10], alpha=0.5)
    ax1.set_title('Texto Original', fontsize=12)
    ax1.set_xlabel('Frecuencia', fontsize=12)
    ax1.set_ylabel('Palabra', fontsize=12)
    ax1.invert_yaxis()
    ax1.set_yticklabels(list(tokens_df['token'][0:10].values))
    # ax1.set_xticklabels(list(tokens_df['Count'][0:10].values))
    ax1.set_xticklabels(['{:,.0f}'.format(x) + 'k'
                         for x in ax1.get_xticks() / 1000])
    ax2.barh(tokens_no_sw_df['token'][1:11], tokens_no_sw_df['Count'][1:11],
             alpha=0.5)
    ax2.set_title('Sin Stopwords', fontsize=12)
    ax2.set_xlabel('Frecuencia', fontsize=12)
    ax2.invert_yaxis()
    ax2.set_yticklabels(list(tokens_no_sw_df['token'][1:11].values))
    # ax2.set_xticklabels(list(tokens_no_sw_df['Count'][1:11].values))
    ax2.set_xticklabels(['{:,.0f}'.format(x) + 'k'
                         for x in ax2.get_xticks() / 1000])
    plt.tight_layout()
    fig.savefig("images/palabrasfreq.png")

    # Palabras de baja frecuencia
    baja_frec = tokens_no_sw_df[tokens_no_sw_df['Count'] == 1].sort_values(by='token')

    # Volumen por fecha
    print("    *Tuits por fecha")
    senti_dia = data.groupby(['date'], as_index=False).agg(sum)
    m_dia = round(senti_dia['cuenta']
                  [senti_dia['date'] != datetime.date(2019, 9, 30)].mean())
    me_dia = round(senti_dia['cuenta']
                   [senti_dia['date'] != datetime.date(2019, 9, 30)].median())
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(senti_dia['date'][senti_dia['date'] != datetime.date(2019, 9, 30)],
            senti_dia['cuenta'][senti_dia['date'] != datetime.date(2019, 9, 30)])
    ax.plot(senti_dia['date'][senti_dia['date'] != datetime.date(2019, 9, 30)],
            [m_dia] * (len(senti_dia) - 1),
            color="r", label="Media: " + '{:,}'.format(m_dia))
    ax.plot(senti_dia['date'][senti_dia['date'] != datetime.date(2019, 9, 30)],
            [me_dia] * (len(senti_dia) - 1),
            color="g", label="Mediana: " + '{:,}'.format(me_dia))
    ax.set_title('Tuits por fecha', fontsize=12)
    ax.set_xlabel('Fecha', size=12)
    ax.set_ylabel('Volumen', size=12)
    ax.set_facecolor('white')
    plt.legend(fontsize=12, loc='lower right')
    fig.savefig("images/tuits_fechas.png")

    # Volumen por horario
    print("    *Tuits por horario")
    senti_hora = data.groupby(['time_3'], as_index=False).agg({'cuenta': sum})
    m_hora = round(senti_hora['cuenta'].mean())
    me_hora = round(senti_hora['cuenta'].median())
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(senti_hora['time_3'], senti_hora['cuenta'])
    ax.plot(senti_hora['time_3'],
            [m_hora] * (len(senti_hora)),
            color="r", label="Media: " + '{:,}'.format(m_hora))
    ax.plot(senti_hora['time_3'],
            [me_hora] * (len(senti_hora)),
            color="g", label="Mediana: " + '{:,}'.format(me_hora))
    ax.set_title('Tuits por hora', fontsize=12)
    ax.set_xlabel('Hora', size=12)
    ax.set_ylabel('Volumen', size=12)
    ax.set_facecolor('white')
    plt.legend(fontsize=12, loc='lower right')
    fig.savefig("images/tuits_horarios.png")

    # Volumen por día de la semana
    print("    *Tuits por día de la semana")
    senti_sem = data.groupby(['weekday'], as_index=False).agg(sum)
    m_sem = round(senti_sem['cuenta'].mean())
    me_sem = round(senti_sem['cuenta'].median())
    senti_sem['weekday'] = pd.Categorical(senti_sem['weekday'],
                                          ['Lunes', 'Martes', 'Miércoles',
                                           'Jueves', 'Viernes', 'Sábado', 'Domingo'])
    senti_sem.sort_values('weekday', inplace=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(senti_sem['weekday'], senti_sem['cuenta'])
    ax.plot(senti_sem['weekday'],
            [m_sem] * (len(senti_sem)),
            color="r", label="Media: " + '{:,}'.format(m_sem))
    ax.plot(senti_sem['weekday'],
            [me_sem] * (len(senti_sem)),
            color="g", label="Mediana: " + '{:,}'.format(me_sem))
    ax.set_title('Tuits por día de la semana', fontsize=12)
    ax.set_xlabel('Día de la semana', size=12)
    ax.set_ylabel('Volumen', size=12)
    ax.set_facecolor('white')
    plt.legend(fontsize=12, loc='lower right')
    fig.savefig("images/tuits_semana.png")

    # Subplots de variables construidas
    print("    *Palabras")
    fig, ax = plt.subplots(figsize=(10, 7), dpi=80, facecolor='white')
    sns.distplot(data['Num_Palabras'], vertical=False)
    plt.axvline(data['Num_Palabras'].mean(), color='g', linestyle='--',
                label='Media: ' + str(round(data['Num_Palabras'].mean(), 1)))
    plt.axvline(data['Num_Palabras'].median(), color='r', linestyle='--',
                label='Mediana: ' + str(round(data['Num_Palabras'].median(), 1)))
    plt.title('Número de palabras', fontsize=16)
    plt.xlabel('Palabras', size=14)
    plt.ylabel('Densidad', size=14)
    # plt.legend({'Media': data['Num_Palabras'].mean(),
    #            'Mediana':data['Num_Palabras'].median()},
    #           prop={'size': 12})
    plt.legend(prop={'size': 14})
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    # plt.grid(False)
    # fig.set_facecolor('white')
    # fig.patch.set_facecolor('white')
    fig.savefig("images/hist_num_palabras.png")

    print("    *Caracteres")
    fig, ax = plt.subplots(figsize=(10, 7), dpi=80, facecolor='white')
    sns.distplot(data['Num_Caraceteres'], vertical=False)
    plt.axvline(data['Num_Caraceteres'].mean(), color='g', linestyle='--',
                label='Media: ' + str(round(data['Num_Caraceteres'].mean(), 1)))
    plt.axvline(data['Num_Caraceteres'].median(), color='r', linestyle='--',
                label='Mediana: ' + str(round(data['Num_Caraceteres'].median(), 1)))
    plt.title('Número de caracteres', fontsize=16)
    plt.xlabel('Caracteres', size=14)
    plt.ylabel('Densidad', size=14)
    plt.legend(prop={'size': 14})
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    # plt.grid(False)
    # fig.set_facecolor('white')
    # fig.patch.set_facecolor('white')
    fig.savefig("images/hist_num_caracteres.png")

    print("    *Palabras de alto")
    fig, ax = plt.subplots(figsize=(10, 7), dpi=80, facecolor='white')
    sns.distplot(data['Num_Stopwords'], vertical=False)
    plt.axvline(data['Num_Stopwords'].mean(), color='g', linestyle='--',
                label='Media: ' + str(round(data['Num_Stopwords'].mean(), 1)))
    plt.axvline(data['Num_Stopwords'].median(), color='r', linestyle='--',
                label='Mediana: ' + str(round(data['Num_Stopwords'].median(), 1)))
    plt.title('Número de Stopwords', fontsize=16)
    plt.xlabel('Stopwords', size=14)
    plt.ylabel('Densidad', size=14)
    plt.legend(prop={'size': 14})
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    # plt.grid(False)
    # fig.set_facecolor('white')
    # fig.patch.set_facecolor('white')
    fig.savefig("images/hist_num_stopwords.png")

    print("    *Largo de palabras")
    plt.figure(figsize=(10, 7), dpi=80, facecolor='white')
    sns.distplot(data['Largo_promedio'][data['Largo_promedio'] <= 20],
                 vertical=False)
    plt.axvline(data['Largo_promedio'][data['Largo_promedio'] <= 20].mean(),
                color='g', linestyle='--',
                label='Media: ' + str(round(data['Largo_promedio']
                                            [data['Largo_promedio'] <= 20].mean(), 1)))
    plt.axvline(data['Largo_promedio'][data['Largo_promedio'] <= 20].median(),
                color='r', linestyle='--',
                label='Mediana: ' + str(round(data['Largo_promedio']
                                              [data['Largo_promedio'] <= 20].median(), 1)))
    plt.title('Extensión promedio de palabras', fontsize=16)
    plt.xlabel('Extensión promedio de palabras', size=14)
    plt.ylabel('Densidad', size=14)
    plt.legend(prop={'size': 14})
    # plt.grid(False)
    # fig.set_facecolor('white')
    # fig.patch.set_facecolor('white')
    fig.savefig("images/hist_larg_promedio.png")

    print("    *Retweets")
    fig, ax = plt.subplots(figsize=(10, 7), dpi=80, facecolor='white')
    sns.distplot(data['nbr_retweet'][data['nbr_retweet'] < 1000], vertical=False)
    plt.axvline(data['nbr_retweet'][data['nbr_retweet'] < 1000].mean(),
                color='g', linestyle='--',
                label='Media: ' + str(round(data['nbr_retweet']
                                            [data['nbr_retweet'] <= 1000].mean(), 1)))
    plt.axvline(data['nbr_retweet'][data['nbr_retweet'] < 1000].median(),
                color='r', linestyle='--',
                label='Mediana: ' + str(round(data['nbr_retweet']
                                              [data['nbr_retweet'] <= 1000].median(), 1)))
    plt.title('Número de $\it{Retweets}$', fontsize=16)
    plt.xlabel('$\it{Retweets}$', size=14)
    plt.ylabel('Densidad', size=14)
    plt.legend(prop={'size': 14})
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    # plt.grid(False)
    # fig.set_facecolor('white')
    # fig.patch.set_facecolor('white')
    fig.savefig("images/hist_num_retweets.png")

    print("    *Respuestas")
    fig, ax = plt.subplots(figsize=(10, 7), dpi=80, facecolor='white')
    sns.distplot(data['nbr_reply'][data['nbr_reply'] > 0], vertical=False)
    plt.axvline(data['nbr_reply'][data['nbr_reply'] > 0].mean(),
                color='g', linestyle='--',
                label='Media: ' + str(round(data['nbr_reply']
                                            [data['nbr_reply'] <= 1000].mean(), 1)))
    plt.axvline(data['nbr_reply'][data['nbr_reply'] > 0].median(),
                color='r', linestyle='--',
                label='Mediana: ' + str(round(data['nbr_reply']
                                              [data['nbr_reply'] <= 1000].median(), 1)))
    plt.title('Número de Respuestas', fontsize=16)
    plt.xlabel('Respuestas', size=14)
    plt.ylabel('Densidad', size=14)
    plt.legend(prop={'size': 14})
    # ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    # plt.grid(False)
    # fig.set_facecolor('white')
    # fig.patch.set_facecolor('white')
    fig.savefig("images/hist_num_respuestas.png")

    # BOXPLOTS DE VARIABLES DE CONTEXTO DE TWITTER (VERTICAL)
    print("    *Variables de contexto - vertical")
    fig, axs = plt.subplots(1, 5, figsize=(30, 10))  # , sharey=True)
    hashtags_plt = axs[0].boxplot(data['Num_Hashtags'][data['Num_Hashtags'] > 0],
                                  flierprops={'markerfacecolor': '#cedbe8',
                                              'markeredgecolor': '#cedbe8'},
                                  patch_artist=True,
                                  # labels=['Hashtags'],
                                  showmeans=True,
                                  meanline=True,
                                  widths=0.9)
    axs[0].set_title('$\it{Hashtags} "#"', fontsize=30)
    axs[0].tick_params(labelsize=20)
    axs[0].set_facecolor('white')
    hashtags_plt['boxes'][0].set(facecolor='#cedbe8')
    plt.setp(axs[0].get_xticklabels(), visible=False)
    axs[0].legend([hashtags_plt["means"][0], hashtags_plt["medians"][0]],
                  ['Media', 'Mediana'], loc='center right',
                  fontsize='large')
    cashtags_plt = axs[1].boxplot(data['Num_Tickers'][data['Num_Tickers'] > 0],
                                  flierprops={'markerfacecolor': '#82a9c5',
                                              'markeredgecolor': '#82a9c5'},
                                  patch_artist=True,
                                  # labels=['Cashtags'],
                                  showmeans=True,
                                  meanline=True,
                                  widths=0.9)
    axs[1].set_title('$\it{Cashtags} "$"', fontsize=30)
    axs[1].tick_params(labelsize=20)
    axs[1].grid(False)
    axs[1].set_facecolor('white')
    cashtags_plt['boxes'][0].set(facecolor='#82a9c5')
    plt.setp(axs[1].get_xticklabels(), visible=False)
    axs[1].legend([cashtags_plt["means"][0], cashtags_plt["medians"][0]],
                  ['Media', 'Mediana'], loc='center right',
                  fontsize='large')
    menciones_plt = axs[2].boxplot(data['Num_Menciones']
                                   [data['Num_Menciones'] > 0],
                                   flierprops={'markerfacecolor': '#41709e',
                                               'markeredgecolor': '#41709e'},
                                   patch_artist=True,
                                   # labels=['Menciones'],
                                   showmeans=True,
                                   meanline=True,
                                   widths=0.9)
    axs[2].set_title('Menciones "@"', fontsize=30)
    axs[2].tick_params(labelsize=20)
    axs[2].grid(False)
    axs[2].set_facecolor('white')
    menciones_plt['boxes'][0].set(facecolor='#41709e')
    plt.setp(axs[2].get_xticklabels(), visible=False)
    axs[2].legend([menciones_plt["means"][0], menciones_plt["medians"][0]],
                  ['Media', 'Mediana'], loc='center right',
                  fontsize='large')
    retweets_plt = axs[3].boxplot(data['nbr_retweet']
                                  [data['nbr_retweet'].between(50, 10000)],
                                  flierprops={'markerfacecolor': '#41709e',
                                              'markeredgecolor': '#41709e'},
                                  patch_artist=True,
                                  # labels=['Retweets'],
                                  showmeans=True,
                                  meanline=True,
                                  widths=0.9)
    axs[3].set_title('Retuits', fontsize=30)
    axs[3].tick_params(labelsize=20)
    axs[3].grid(False)
    axs[3].set_facecolor('white')
    retweets_plt['boxes'][0].set(facecolor='#41709e')
    plt.setp(axs[3].get_xticklabels(), visible=False)
    axs[3].legend([retweets_plt["means"][0], retweets_plt["medians"][0]],
                  ['Media', 'Mediana'], loc='center right',
                  fontsize='large')
    respuestas_plt = axs[4].boxplot(data['nbr_reply']
                                    [data['nbr_reply'].between(50, 10000)],
                                    flierprops={'markerfacecolor': '#41709e',
                                                'markeredgecolor': '#41709e'},
                                    patch_artist=True,
                                    # labels=['Respuestas'],
                                    showmeans=True,
                                    meanline=True,
                                    widths=0.9)
    axs[4].set_title('Respuestas', fontsize=30)
    axs[4].tick_params(labelsize=20)
    axs[4].grid(False)
    axs[4].set_facecolor('white')
    respuestas_plt['boxes'][0].set(facecolor='#41709e')
    plt.setp(axs[4].get_xticklabels(), visible=False)
    axs[4].legend([respuestas_plt["means"][0], respuestas_plt["medians"][0]],
                  ['Media', 'Mediana'], loc='center right',
                  fontsize='large')
    fig.savefig("images/boxplots_sp_char_2.png")

    # BOXPLOTS DE VARIABLES DE CONTEXTO DE TWITTER (HORIZONTAL)
    print("    *Variables de contexto - horizontal")
    medianprops = dict(linestyle='-', linewidth=3, color='red')
    meanprops = dict(linestyle='-', linewidth=3, color='green')

    fig, axs = plt.subplots(5, 1, figsize=(10, 12))
    hashtags_plt = axs[0].boxplot(data['Num_Hashtags'][data['Num_Hashtags'] > 0],
                                  flierprops={'markerfacecolor': '#82a9c5',
                                              'markeredgecolor': '#82a9c5'},
                                  patch_artist=True,
                                  # labels=['Hashtags'],
                                  # boxprops=dict(facecolor='red'),
                                  medianprops=medianprops,
                                  meanprops=meanprops,
                                  showmeans=True,
                                  meanline=True,
                                  widths=0.9,
                                  vert=False)
    axs[0].set_title('$\it{Hashtags}$', fontsize=14)
    # axs[0].tick_params(labelsize=14)
    # axs[0].set_facecolor('white')
    hashtags_plt['boxes'][0].set(facecolor='#82a9c5')
    plt.setp(axs[0].get_yticklabels(), visible=False)
    axs[0].legend([hashtags_plt["means"][0], hashtags_plt["medians"][0]],
                  ['Media', 'Mediana'], loc='center right',
                  fontsize='large')
    cashtags_plt = axs[1].boxplot(data['Num_Tickers'][data['Num_Tickers'] > 0],
                                  flierprops={'markerfacecolor': '#82a9c5',
                                              'markeredgecolor': '#82a9c5'},
                                  patch_artist=True,
                                  # labels=['Cashtags'],
                                  medianprops=medianprops,
                                  meanprops=meanprops,
                                  showmeans=True,
                                  meanline=True,
                                  widths=0.9,
                                  vert=False)
    axs[1].set_title('$\it{Cashtags}$', fontsize=14)
    # axs[1].tick_params(labelsize=20)
    axs[1].grid(False)
    axs[1].set_facecolor('white')
    cashtags_plt['boxes'][0].set(facecolor='#82a9c5')
    plt.setp(axs[1].get_yticklabels(), visible=False)
    axs[1].legend([cashtags_plt["means"][0], cashtags_plt["medians"][0]],
                  ['Media', 'Mediana'], loc='center right',
                  fontsize='large')
    menciones_plt = axs[2].boxplot(data['Num_Menciones'][data['Num_Menciones'] > 1],
                                   flierprops={'markerfacecolor': '#82a9c5',
                                               'markeredgecolor': '#82a9c5'},
                                   patch_artist=True,
                                   # labels=['Menciones'],
                                   medianprops=medianprops,
                                   meanprops=meanprops,
                                   showmeans=True,
                                   meanline=True,
                                   widths=0.9,
                                   vert=False)
    axs[2].set_title('Menciones', fontsize=14)
    # axs[2].tick_params(labelsize=20)
    axs[2].grid(False)
    axs[2].set_facecolor('white')
    menciones_plt['boxes'][0].set(facecolor='#82a9c5')
    plt.setp(axs[2].get_yticklabels(), visible=False)
    axs[2].set_xticklabels(['{:,.0f}'.format(x) for x in axs[2].get_xticks()])
    axs[2].legend([menciones_plt["means"][0], menciones_plt["medians"][0]],
                  ['Media', 'Mediana'], loc='center right',
                  fontsize='large')
    retweets_plt = axs[3].boxplot(data['nbr_retweet']
                                  [data['nbr_retweet'].between(50, 10000)],
                                  flierprops={'markerfacecolor': '#82a9c5',
                                              'markeredgecolor': '#82a9c5'},
                                  patch_artist=True,
                                  # labels=['Retweets'],
                                  medianprops=medianprops,
                                  meanprops=meanprops,
                                  showmeans=True,
                                  meanline=True,
                                  widths=0.9,
                                  vert=False)
    axs[3].set_title('Retuits', fontsize=14)
    # axs[3].tick_params(labelsize=20)
    axs[3].grid(False)
    axs[3].set_facecolor('white')
    retweets_plt['boxes'][0].set(facecolor='#82a9c5')
    plt.setp(axs[3].get_yticklabels(), visible=False)
    axs[3].legend([retweets_plt["means"][0], retweets_plt["medians"][0]],
                  ['Media', 'Mediana'], loc='center right',
                  fontsize='large')
    respuestas_plt = axs[4].boxplot(data['nbr_reply']
                                    [data['nbr_reply'].between(50, 10000)],
                                    flierprops={'markerfacecolor': '#82a9c5',
                                                'markeredgecolor': '#82a9c5'},
                                    patch_artist=True,
                                    # labels=['Respuestas'],
                                    medianprops=medianprops,
                                    meanprops=meanprops,
                                    showmeans=True,
                                    meanline=True,
                                    widths=0.9,
                                    vert=False)
    axs[4].set_title('Respuestas', fontsize=14)
    # axs[4].tick_params(labelsize=20)
    axs[4].grid(False)
    axs[4].set_facecolor('white')
    respuestas_plt['boxes'][0].set(facecolor='#82a9c5')
    plt.setp(axs[4].get_yticklabels(), visible=False)
    axs[4].legend([respuestas_plt["means"][0], respuestas_plt["medians"][0]],
                  ['Media', 'Mediana'], loc='center right',
                  fontsize='large')
    plt.tight_layout()
    plt.savefig("images/boxplots_sp_char_3.png")

    # PAIRPLOT DE LA RELACIÓN BIVARIADA DE LOS DATOS
    print("    *Relación bivariada")
    leo_1 = pd.read_excel(CLAS_MANUAL_2)
    # leo_1.reset_index(inplace=True)
    # leo_1.rename(columns={'index': 'text', 'content': 'polarity'}, inplace=True)
    leo_2 = pd.read_excel(CLAS_MANUAL_3)
    tweets_veamos = pd.concat([leo_1,
                               leo_2],
                              axis=0, ignore_index=True)
    data_test = data[data['text'].isin(tweets_veamos['content'])]
    data_pair = data_test[['Num_Palabras', 'Num_Caraceteres',
                           'Num_Stopwords', 'Largo_promedio',
                           'Num_Hashtags', 'Num_Menciones', 'Sentimiento']]
    data_pair = data_pair[data_pair['Sentimiento'].isin(['P', 'N'])]
    data_pair = data_pair.sample(1000)
    sns_pairplot = sns.pairplot(data_pair, hue='Sentimiento')
    sns_pairplot.savefig("images/features_pairplot.png")

    # =============================================================================
    # MATRIZ DE CORRELACIÓN DE VARIABLES
    # =============================================================================
    print("    *Matriz de correlación")
    dict_sent = {'N': -1,
                 'P': 1,
                 'NEU': 0}
    data['Sentimiento'] = data['Sentimiento'].apply(lambda x: dict_sent[x])
    data.rename(columns={'Num_Palabras': 'Palabras',
                         'Num_Caraceteres': 'Caracteres',
                         'Num_Stopwords': 'Stopwords',
                         'Largo_promedio': 'Largo Palabras',
                         'Num_Hashtags': 'Hashtags',
                         'Num_Menciones': 'Menciones',
                         'Num_Tickers': 'Tickers'}, inplace=True)
    ax = plt.axes()
    corr_plot = sns.heatmap(data[['Sentimiento', 'Palabras',
                                  'Caracteres', 'Stopwords',
                                  'Largo Palabras', 'Hashtags',
                                  'Menciones', 'Tickers']].corr(),
                            annot=True,
                            cbar=False, linewidths=1,
                            square=True,
                            # cmap='vlag',
                            cmap=sns.diverging_palette(220, 20, as_cmap=True),
                            fmt='.2f')
    ax.set_title('Matriz de correlación de variables', fontsize=12)
    plt.savefig('images/matriz_corr_variables.png')
