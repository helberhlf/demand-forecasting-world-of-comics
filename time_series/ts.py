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

# Decomposition of the time series
# https://stackoverflow.com/questions/39400115/python-pandas-group-by-date-using-datetime-data
def decomposition(df, date, col):
    # Convert date column to index
    df.index = pd.to_datetime(df[date], format='%Y-%m-%d %H:%M:%S')

    # Decomposition of time series by year, month, week, day and hour on total sales
    df['total_sales_per_year']  = df[col].resample('Y').sum()
    df['total_sales_per_month'] = df[col].resample('MS').sum()
    df['total_sales_per_week']  = df[col].resample('W').sum()
    df['total_sales_per_day']   = df[col].resample('D').sum()
    df['total_sales_per_hour']  = df[col].resample('H').sum()

    # Decomposition of the time series by year, month, week, day and hour on average sales
    df['avg_sales_per_year']  = df[col].resample('Y').mean()
    df['avg_sales_per_month'] = df[col].resample('MS').mean()
    df['avg_sales_per_week']  = df[col].resample('W').mean()
    df['avg_sales_per_day']   = df[col].resample('D').mean()
    df['avg_sales_per_hour']  = df[col].resample('H').mean()

    return df

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

# Function to test stationarity
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

    # Test
    dfteste = adfuller(serie, autolag='AIC')

    # Formatting the output
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