import os
import pandas as pd
import seaborn as sns
import datetime
from plotly.offline import plot
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.ticker as mtick
print("Directorio actual: " + os.getcwd())

# =============================================================================
# INPUTS
# =============================================================================
INPUT_SUP_TUITS = 'data/supervised_tweets.csv'
INPUT_BBAJIOO = 'data/STOCK_prices/BBAJIOO_2018_2020.csv'
INPUT_BSMXB = 'data/STOCK_prices/BSMXB_2018_2020.csv'
INPUT_GENTERA = 'data/STOCK_prices/GENTERA_2018_2020.csv'
INPUT_GFINBURO = 'data/STOCK_prices/GFINBURO_2018_2020.csv'
INPUT_GFNORTEO = 'data/STOCK_prices/GFNORTEO_2018_2020.csv'

# =============================================================================
# FUNCIONES
# =============================================================================


def values(x):
    val = x[-1]
    if val == 'M':
        return float(x[:-1])*1000000
    elif val == 'K':
        return float(x[:-1])*1000
    else:
        return float(x)


# =============================================================================

if __name__ == "__main__":
    # INFORMACIÓN DE TWITTER
    print("Leyendo información de opiniones")
    data = pd.read_csv(INPUT_SUP_TUITS, sep=',', encoding='latin-1')
    data['datetime'] = pd.to_datetime(data['datetime'], errors='coerce')
    data = data[(data['datetime']
                 >= datetime.datetime(2018, 10, 1)) &
                (data['datetime']
                 <= datetime.datetime(2019, 9, 30))].sort_values(by=['datetime'])
    data['date'] = data['datetime'].dt.date
    data['time'] = data['datetime'].dt.time
    data['time_2'] = data['time'].apply(lambda x: datetime.time(x.hour, x.minute, 0, 0))
    data['time_2'] = data['time_2']\
                     .apply(lambda x: datetime.time(x.hour,
                                                    int(round(x.minute, -1)) % 60,
                                                    0, 0))
    # data = data[(data['time'] >= datetime.time(8, 30, 00))
    #             & (data['time'] <= datetime.time(14, 30, 00))]
    senti_dia = data.groupby(['date'], as_index=False).agg(sum)
    senti_hora = data.groupby(['time_2'], as_index=False).agg(sum)
    # INFORMACIÓN FINANCIERA
    print("Leyendo información financiera")
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
        ind_dec = tickers_dict[ticker]['Apertura']-tickers_dict[ticker]['Cierre']
        tickers_dict[ticker]['ticker'] = str(ticker)
        tickers_dict[ticker]['INDICATOR'] = ind_dec.apply(lambda x: -1 if x > 0 else 1)
        tickers_dict[ticker]['Fecha'] = pd.to_datetime(tickers_dict[ticker]['Fecha'],
                                                       errors='coerce',
                                                       format='%d.%m.%Y')
        tickers_dict[ticker]['Fecha'] = tickers_dict[ticker]['Fecha'].dt.date
        # tickers_dict[ticker]['Volumen'] = tickers_dict[ticker]['Vol.'].apply(lambda x:
        #                                                                      values(x))
        tickers_dict[ticker]['Variacion(%)'] = tickers_dict[ticker]['% var.'].apply(lambda x: float(x[:-1]))
        # tickers_dict[ticker].drop(['Vol.', '% var.'], axis=1, inplace=True)
        tickers_dict[ticker] = tickers_dict[ticker].sort_values(by=['Fecha']).reset_index(drop=True)
    tickers_data = pd.DataFrame()
    for ticker in tickers_dict.keys():
        tickers_data = pd.concat([tickers_data, tickers_dict[ticker]],
                                 axis=0, sort=False, ignore_index=True)
    tickers_data = tickers_data[tickers_data['Fecha'].between(datetime.date(2018, 10, 1),
                                                              datetime.date(2019, 9, 30))]
    dias_actividad = list(tickers_dict['BBAJIO']['Fecha'].unique())

    print("Generando gráficos")
    # HISTÓRICO DE PRECIO DE CIERRE
    print("    *Histórico de cierre de precios")
    fig, ax = plt.subplots(5, 1, squeeze=False, sharex=True, figsize=(40, 40))
    fig.suptitle('Precio', fontsize=40, y=0.92, x=0.65)
    ax[0][0].plot(tickers_data[tickers_data['ticker'] == 'BBAJIO']['Fecha'],
                  tickers_data[tickers_data['ticker'] == 'BBAJIO']['Cierre'],
                  linewidth=3)
    ax[0][0].set_title('BBAJIO', fontsize=30)
    ax[0][0].set_facecolor('white')
    ax[0][0].tick_params(axis='y', labelsize=30)
    ax[0][0].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    ax[1][0].plot(tickers_data[tickers_data['ticker'] == 'BSMX']['Fecha'],
                  tickers_data[tickers_data['ticker'] == 'BSMX']['Cierre'],
                  linewidth=3)
    ax[1][0].set_title('BSMX', fontsize=30)
    ax[1][0].set_facecolor('white')
    ax[1][0].tick_params(axis='y', labelsize=30)
    ax[2][0].plot(tickers_data[tickers_data['ticker'] == 'GENTERA']['Fecha'],
                  tickers_data[tickers_data['ticker'] == 'GENTERA']['Cierre'],
                  linewidth=3)
    ax[2][0].set_title('GENTERA', fontsize=30)
    ax[2][0].set_ylabel('Precio', fontsize=30)
    ax[2][0].set_facecolor('white')
    ax[2][0].tick_params(axis='y', labelsize=30)
    ax[3][0].plot(tickers_data[tickers_data['ticker'] == 'GFINBU']['Fecha'],
                  tickers_data[tickers_data['ticker'] == 'GFINBU']['Cierre'],
                  linewidth=3)
    ax[3][0].set_title('GFINBU', fontsize=30)
    ax[3][0].set_facecolor('white')
    ax[3][0].tick_params(axis='y', labelsize=30)
    ax[4][0].plot(tickers_data[tickers_data['ticker'] == 'GFNORT']['Fecha'],
                  tickers_data[tickers_data['ticker'] == 'GFNORT']['Cierre'],
                  linewidth=3)
    ax[4][0].set_title('GFNORT', fontsize=30)
    ax[4][0].set_facecolor('white')
    ax[4][0].tick_params(axis='y', labelsize=30)
    ax[4][0].set_xlabel('Fecha', fontsize=30)
    # plt.xticks(rotation=90)
    plt.subplots_adjust(left=0.4, bottom=0.4)
    plt.tick_params(axis='both', labelsize=30)
    fig.savefig("images/precios_historicos.png")

    # HISTÓRICO DE VARIACION DE PRECIO DE CIERRE
    print("    *Histórico de variación de precios")
    fig, ax = plt.subplots(5, 1, squeeze=False, sharex=True, figsize=(40, 40))
    fig.suptitle('Variación histórica del precio', fontsize=40, y=0.92, x=0.65)
    ax[0][0].plot(tickers_data[tickers_data['ticker'] == 'BBAJIO']['Fecha'],
                  tickers_data[tickers_data['ticker'] == 'BBAJIO']['Variacion(%)'],
                  linewidth=3)
    ax[0][0].set_title('BBAJIO', fontsize=30)
    ax[0][0].set_facecolor('white')
    ax[0][0].tick_params(axis='y', labelsize=30)
    ax[0][0].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    ax[1][0].plot(tickers_data[tickers_data['ticker'] == 'BSMX']['Fecha'],
                  tickers_data[tickers_data['ticker'] == 'BSMX']['Variacion(%)'],
                  linewidth=3)
    ax[1][0].set_title('BSMX', fontsize=30)
    ax[1][0].set_facecolor('white')
    ax[1][0].tick_params(axis='y', labelsize=30)
    ax[2][0].plot(tickers_data[tickers_data['ticker'] == 'GENTERA']['Fecha'],
                  tickers_data[tickers_data['ticker'] == 'GENTERA']['Variacion(%)'],
                  linewidth=3)
    ax[2][0].set_title('GENTERA', fontsize=30)
    ax[2][0].set_ylabel('Variación porcentual', fontsize=30)
    ax[2][0].set_facecolor('white')
    ax[2][0].tick_params(axis='y', labelsize=30)
    ax[3][0].plot(tickers_data[tickers_data['ticker'] == 'GFINBU']['Fecha'],
                  tickers_data[tickers_data['ticker'] == 'GFINBU']['Variacion(%)'],
                  linewidth=3)
    ax[3][0].set_title('GFINBU', fontsize=30)
    ax[3][0].set_facecolor('white')
    ax[3][0].tick_params(axis='y', labelsize=30)
    ax[4][0].plot(tickers_data[tickers_data['ticker'] == 'GFNORT']['Fecha'],
                  tickers_data[tickers_data['ticker'] == 'GFNORT']['Variacion(%)'],
                  linewidth=3)
    ax[4][0].set_title('GFNORT', fontsize=30)
    ax[4][0].set_facecolor('white')
    ax[4][0].tick_params(axis='y', labelsize=30)
    ax[4][0].set_xlabel('Fecha', fontsize=30)
    # plt.xticks(rotation=90)
    plt.subplots_adjust(left=0.4, bottom=0.4)
    plt.tick_params(axis='both', labelsize=30)
    fig.savefig("images/variacion_historica.png")

    # COEFICIENTE DE CORRELACIÓN DE PEARSON
    print("    *Correlación de símbolos")
    corr_tickers = tickers_data.pivot_table(index='Fecha', columns='ticker',
                                            values='Cierre', aggfunc='sum')
    ax = plt.axes()
    corr_plot = sns.heatmap(corr_tickers.corr(), annot=True,
                            cbar=False, linewidths=1,
                            square=True, cmap="Blues")
    ax.set_title('Matriz de correlación de precios', fontsize=16)
    plt.savefig('images/matriz_corr_tickers.png')

    # PROMEDIO PONDERADO DE SÍMBOLOS POR CAAPITALIZACIÓN DE MERCADO
    # w_b = Num de acciones al 31/12/2018 * Promedio de precio de cierre del 2018
    tickers_cap_dict = {'BBAJIO': {'Acciones': 1189931000,
                                   'Precio_prom': tickers_dict['BBAJIO']
                                   ['Cierre'].mean()},
                        'BSMX': {'Acciones': 6786994000,
                                 'Precio_prom': tickers_dict['BSMX']
                                 ['Cierre'].mean()},
                        'GENTERA': {'Acciones': 1605723000,
                                    'Precio_prom': tickers_dict['GENTERA']
                                    ['Cierre'].mean()},
                        'GFINBU': {'Acciones': 6639780000,
                                   'Precio_prom': tickers_dict['GFINBU']
                                   ['Cierre'].mean()},
                        'GFNORT': {'Acciones': 2883456000,
                                   'Precio_prom': tickers_dict['GFNORT']
                                   ['Cierre'].mean()}}
    total = 0
    for ticker in tickers_cap_dict.keys():
        total += tickers_cap_dict[ticker]['Acciones']\
                 * tickers_cap_dict[ticker]['Precio_prom']
    for ticker in tickers_cap_dict.keys():
        tickers_cap_dict[ticker]['Weight'] = tickers_cap_dict[ticker]['Acciones']\
                                             * tickers_cap_dict[ticker]['Precio_prom']/total
    """
    GENTERA = 3.36%
    BBAJIO = 6.11%
    BSMX = 23.72%
    GFINBU = 24.89%
    GFNORT = 41.93%
    """

    # INDICADORES DE MODELO
    df_dias_actividad = pd.DataFrame({'Fecha': dias_actividad})
    df_dias_actividad.reset_index(inplace=True)
    poc_ant = tickers_data.groupby('Fecha', as_index=False).agg({'Cierre': 'sum'})
    poc_ant_copy = poc_ant.copy()
    poc_ant['Fecha_Ant'] = ''
    for i, fecha in enumerate(poc_ant['Fecha']):
        indice = df_dias_actividad[df_dias_actividad['Fecha'] == fecha]['index'].iloc[0]
        poc_ant['Fecha_Ant'].iloc[i] = df_dias_actividad[df_dias_actividad['index']
                                                         == indice-1]['Fecha'].iloc[0]
    poc = poc_ant.merge(poc_ant_copy, how='left', left_on='Fecha_Ant', right_on='Fecha')
    poc.drop(['Fecha_y'], axis=1, inplace=True)
    poc.rename(columns={'Fecha_x': 'Fecha_Actual',
                        'Cierre_x': 'Cierre_Actual',
                        'Cierre_y': 'Cierre_Ant'}, inplace=True)
    poc['Variacion'] = (poc['Cierre_Actual']/poc['Cierre_Ant'])-1
    del poc_ant, poc_ant_copy, df_dias_actividad, fecha, i, indice

    # INDICADOR DE COMPRA
    print("    *Indicador de compra")
    fig = plt.figure(num=None, figsize=(6, 5))
    ax = poc['Variacion'].hist(bins=40, color='#A5C8E1')
    plt.axvline(x=0.005, ymin=0, ymax=1, linestyle='--',
                color='green', lw=2, label='0.3%')
    plt.legend(loc='upper right', prop={'size': 10})
    ax.set_facecolor('white')
    ax.grid(False)
    ax.set_title('% de Variación diario del sector financiero', fontsize=15)
    ax.set_facecolor('white')
    ax.tick_params(axis='y', labelsize=5)
    ax.set_ylabel('Frecuencia', fontsize=10)
    ax.tick_params(axis='x', labelsize=10)
    fmt = '%.2f%%'  # Formato de los ticks, p.e. '40%'
    xticks = mtick.FormatStrFormatter(fmt)
    ax.xaxis.set_major_formatter(xticks)
    ax.set_xlabel('Variacion', fontsize=10)
    fig.savefig("images/hist_variacion_tickers.png")

    # INDICADOR DE VENTA
    print("    *Indicador de venta")
    fig = plt.figure(num=None, figsize=(15, 10))
    ax = poc['Variacion'].hist(bins=40, color='#A5C8E1')
    plt.axvline(x=-0.005, ymin=0, ymax=1, linestyle='--',
                color='red', lw=4, label='0.5%')
    plt.legend(loc='upper right', prop={'size': 25})
    ax.set_facecolor('white')
    ax.grid(False)
    ax.set_title('% de Variación diario del sector financiero', fontsize=30)
    ax.set_facecolor('white')
    ax.tick_params(axis='y', labelsize=20)
    ax.set_ylabel('Frecuencia', fontsize=30)
    ax.tick_params(axis='x', labelsize=20)
    fmt = '%.2f%%'
    xticks = mtick.FormatStrFormatter(fmt)
    ax.xaxis.set_major_formatter(xticks)
    ax.set_xlabel('Variacion', fontsize=30)
    fig.savefig("images/hist_neg_variacion_tickers.png")
