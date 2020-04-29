import numpy as np
from itertools import zip_longest
import pandas as pd
import matplotlib.pyplot as plt
import csv
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
import datetime

plt.style.use('seaborn')

confirmed_cases = pd.read_csv('https://raw.github.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
deaths_reported = pd.read_csv('https://raw.github.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
recovered_cases = pd.read_csv('https://raw.github.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')

print(confirmed_cases.head())
cols = confirmed_cases.keys()

india_confirmed = confirmed_cases[confirmed_cases['Country/Region'] == 'India']
india_deaths = deaths_reported[deaths_reported['Country/Region'] == 'India']
india_recoveries = recovered_cases[recovered_cases['Country/Region'] == 'India']

cols = confirmed_cases.keys()

confirmed_india = india_confirmed.loc[:,cols[4]:cols[-1]]
deaths_india = india_deaths.loc[:,cols[4]:cols[-1]]
recovered_india = india_recoveries.loc[:,cols[4]:cols[-1]]

dates = confirmed_india.keys()
india_cases = []
india_deaths = []
india_recovered = []
for i in dates:
    confirmed_sum = confirmed_india[i].sum()
    death_sum = deaths_india[i].sum()
    recovered_sum = recovered_india[i].sum()
    india_cases.append(confirmed_sum)
    india_deaths.append(death_sum)
    india_recovered.append(recovered_sum)


print("Total cases in India : ", confirmed_sum)
print("Total deaths in India : ", death_sum)
print("Total recovered cases in India : ", recovered_sum)

days = []
future_forecast =[i for i in range(len(dates))]
adjusted_dates = future_forecast[:-10]
start = '1/22/2020'
start_date = datetime.datetime.strptime(start,'%m/%d/%Y')
print(start_date)

d = [dates, india_cases,india_deaths,india_recovered]
merged_data = zip_longest(*d, fillvalue = '')
with open('india_data.csv', 'w', encoding="ISO-8859-1", newline='') as myfile:
      wr = csv.writer(myfile)
      wr.writerow(("Date", "Cases","Deaths","Recovered"))
      wr.writerows(merged_data)
myfile.close()

data = pd.read_csv('india_data.csv')
data['Date'] = pd.to_datetime(data['Date'], infer_datetime_format=True)
data['Cases'] = data['Cases'] - data['Cases'].shift(1)
data['Deaths'] = data['Deaths'] - data['Deaths'].shift(1)
data['Recovered'] = data['Recovered'] - data['Recovered'].shift(1)
data.fillna(0)
data = data.set_index(data.columns[0])
data = data.sort_index()
print(data.tail())

def dates_generator(start,days):
    value = pd.date_range(start=start, periods=days+1, freq='D', closed='right')
    seven_day_forecast = pd.DataFrame(index=value)
    return seven_day_forecast

def get_column_name_and_value(data,i):
    value = data[[data.columns[i]]].dropna()
    name = data.columns[i]
    return value, name

def train_test_split(value, name, ratio):
    n_row = len(value)
    print(name+' total samples: ',n_row)
    split_row = int((n_row)*ratio)
    print('Training samples: ',split_row)
    print('Testing samples: ',n_row-split_row)
    train = value.iloc[:split_row]
    test = value.iloc[split_row:]
    return train, test, split_row

def transformation_of_data(train,test):
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train)
    test_scaled = scaler.fit_transform(test)
    train_scaled_df = pd.DataFrame(train_scaled, index = train.index, columns=[train.columns[0]])
    test_scaled_df = pd.DataFrame(test_scaled,index = test.index, columns=[test.columns[0]])
    return train_scaled_df, test_scaled_df, scaler

def timeseries_feature_builder(df, lag):
    df_copy = df.copy()
    for i in range(1,lag):
        df_copy['lag'+str(i)] = df.shift(i)
    return df_copy

def create_arrays(train,test):
    X_train_array = train.dropna().drop(train.columns[0], axis=1).values
    y_train_array = train.dropna()[train.columns[0]].values
    X_test_array = test.dropna().drop(test.columns[0], axis=1).values
    y_test_array = test.dropna()[test.columns[0]].values
    return X_train_array, y_train_array, X_test_array, y_test_array

def fit_svr(X_train_array, y_train_array, X_test_array, y_test_array):
    svr = SVR(kernel='rbf', gamma='auto', tol=0.001, C=10.0, epsilon=0.001)
    svr.fit(X_train_array,y_train_array)
    y_pred_train = svr.predict(X_train_array)
    y_pred_test = svr.predict(X_test_array)
    print('r-square_SVR_Test: ', round(svr.score(X_test_array,y_test_array),2))
    return svr, y_pred_test

def svr_result_validator(scaler, y_pred_test, value, split_row, lag):
    new_test = value.iloc[split_row:]
    test_pred = new_test.iloc[lag:].copy()
    y_pred_test_transformed = scaler.inverse_transform([y_pred_test])
    y_pred_test_transformed_reshaped = np.reshape(y_pred_test_transformed,(y_pred_test_transformed.shape[1],-1))
    test_pred['Forecast'] = np.array(y_pred_test_transformed_reshaped)
    return test_pred


def svr_forecast(X_test_array, days, svr, lag, scaler):
    last_test_sample = X_test_array[-1]
    X_last_test_sample = np.reshape(last_test_sample, (-1, X_test_array.shape[1]))
    y_pred_last_sample = svr.predict(X_last_test_sample)
    new_array = X_last_test_sample
    new_predict = y_pred_last_sample

    seven_days_svr = []
    for i in range(0, days):
        new_array = np.insert(new_array, 0, new_predict)
        new_array = np.delete(new_array, -1)
        new_array_reshape = np.reshape(new_array, (-1, lag))
        new_predict = svr.predict(new_array_reshape)
        temp_predict = scaler.inverse_transform([new_predict])
        seven_days_svr.append(temp_predict[0][0].round(2))

    return seven_days_svr


def prediction(data, lag, days):
    seven_day_forecast_svr = dates_generator('2020-04-28', days)

    for i in range(len(data.columns)):
        # preprocessing
        value, name = get_column_name_and_value(data, i)
        train, test, split_row = train_test_split(value, name, 0.80)
        train_scaled_df, test_scaled_df, scaler = transformation_of_data(train, test)
        train = timeseries_feature_builder(train_scaled_df, lag + 1)
        test = timeseries_feature_builder(test_scaled_df, lag + 1)
        X_train_array, y_train_array, X_test_array, y_test_array = create_arrays(train, test)

        # SVR modeling
        svr, y_pred_test = fit_svr(X_train_array, y_train_array, X_test_array, y_test_array)
        test_pred = svr_result_validator(scaler, y_pred_test, value, split_row, lag)
        seven_days_svr = svr_forecast(X_test_array, days, svr, lag, scaler)
        seven_day_forecast_svr[name] = np.array(seven_days_svr)

        # plot result
        plt.figure(figsize=(20, 10))
        plt.plot(test_pred)
        plt.plot(seven_day_forecast_svr[name], color='red', label='Forecast')
        plt.ylabel('Cases Counts')
        plt.legend(loc='upper right')
        plt.title(name + '- next 7 days Forecast')
        plt.show()

    return (seven_day_forecast_svr)

svr_prediction = prediction(data, 4, 4)

print(svr_prediction.head())