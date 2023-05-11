#-------------------------------------------------------
# @author Helber
#-------------------------------------------------------
# Importing library for manipulation and exploration of datasets.
import numpy as np
import pandas as pd

# Imports for time series analysis and modeling
from statsmodels.tsa.stattools import adfuller

#Import library to generate specific sets of holidays
import holidays
import datetime

# Importing libraries needed for matplotlib
import matplotlib.pyplot as plt
#----------------------------------------------------

# Function to list holidays
def list_holidays(df):
    df['date']= pd.to_datetime(df.date)

    min_year = df.date.min().year
    max_year = df.date.max().year

    years_list = pd.period_range(min_year, max_year, freq='Y')

    list_of_holidays = []

    for year in years_list:
        list_of_holidays.append(holidays.BR(years=int(str(year))).keys())

    holiday_list = [item for sublist in list_of_holidays for item in sublist]

    return holiday_list

# Function to create the new date and time attributes
def create_dt_attributes(df):
    df['date'] = pd.to_datetime(df.date)
    df['month'] = df.date.dt.month
    df['day_of_month'] = df.date.dt.day
    df['day_of_year'] = df.date.dt.dayofyear
    df['week_of_year'] = df.date.dt.weekofyear
    df['day_of_week'] = df.date.dt.weekday + 1
    df['year'] = df.date.dt.year
    df['is_weekend'] = df.date.dt.weekday // 5
    df['start_of_month'] = df.date.dt.is_month_start.astype(int)
    df['end_of_month'] = df.date.dt.is_month_end.astype(int)
    df['is_holiday'] = np.where(df.date.isin(list_holidays(df)), 1, 0)

    return df

# Função para testar a estacionaridade
def test_stationarity(serie):
    # Calcula estatísticas móveis
    rolmean = serie.rolling(window=12).mean()
    rolstd = serie.rolling(window=12).std()

    # Plot das estatísticas móveis
    orig = plt.plot(serie, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='Média Móvel')
    std = plt.plot(rolstd, color='black', label='Desvio Padrão')

    # Plot
    plt.legend(loc='best')
    plt.title('Estatísticas Móveis - Média e Desvio Padrão')
    plt.show()

    # Teste Dickey-Fuller:
    # Print
    print('\nResultado do Teste Dickey-Fuller:\n')

    # Teste
    dfteste = adfuller(serie, autolag='AIC')

    # Formatando a saída
    dfsaida = pd.Series(dfteste[0:4], index=['Estatística do Teste',
                                             'Valor-p',
                                             'Número de Lags Consideradas',
                                             'Número de Observações Usadas'])

    # Loop por cada item da saída do teste
    for key, value in dfteste[4].items():
        dfsaida['Valor Crítico (%s)' % key] = value

    # Print
    print(dfsaida)

    # Testa o valor-p
    print('\nConclusão:')
    if dfsaida[1] > 0.05:
        print('\nO valor-p é maior que 0.05 e, portanto, não temos evidências para rejeitar a hipótese nula.')
        print('Essa série provavelmente não é estacionária.')
    else:
        print('\nO valor-p é menor que 0.05 e, portanto, temos evidências para rejeitar a hipótese nula.')
        print('Essa série provavelmente é estacionária.')