import streamlit as stm
import pandas as pd
import numpy as np
import pandas_datareader as pdr
import tensorflow as tf
from tensorflow import keras
import matplotlib as plt
from sklearn.preprocessing import MinMaxScaler
import math
from sklearn.metrics import mean_squared_error
Model=keras.models.load_model('Mfinalmodel.h5')

stm.header("Home")
Organization = stm.selectbox("Organization: ",  ['FB', 'AAPL'])
if(Organization=='AAPL'):
    adf=pd.read_csv('AAPL.csv')
    adf1=adf.reset_index()['close']
    scaler=MinMaxScaler(feature_range=(0,1))
    adf1=scaler.fit_transform(np.array(adf1).reshape(-1,1))
    training_size=int(len(adf1)*0.65)
    test_size=len(adf1)-training_size
    train_data,test_data=adf1[0:training_size,:],adf1[training_size:len(adf1),:1]
    def create_dataset(dataset,time_step=1):
        dataX,dataY=[],[]
        for i in range(len(dataset)-time_step-1):
            a=dataset[i:(i+time_step), 0]
            dataX.append(a)
            dataY.append(dataset[i+time_step,0])
        return np.array(dataX),np.array(dataY)
    time_step=100
    X_train,y_train=create_dataset(train_data,time_step)
    X_test,y_test=create_dataset(test_data,time_step)
    X_train=X_train.reshape(X_train.shape[0],X_train.shape[1],1)
    X_test=X_test.reshape(X_test.shape[0],X_test.shape[1],1)
    train_predict=Model.predict(X_train)
    test_predict=Model.predict(X_test)
    train_predict=scaler.inverse_transform(train_predict)
    test_predict=scaler.inverse_transform(test_predict)
    look_back=100
    trainPredictPlot=np.empty_like(adf1)
    trainPredictPlot[:, :]=np.nan
    trainPredictPlot[look_back:len(train_predict)+look_back, :]=train_predict
    testPredictPlot=np.empty_like(adf1)
    testPredictPlot[:, :]=np.nan
    testPredictPlot[len(train_predict)+(look_back*2)+1:len(adf1)-1:]=test_predict
    plt.plot(scaler.inverse_transform(adf1))
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.show()
    x_input=test_data[341:].reshape(1,-1)
    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    lst_output=[]
    n_steps=100
    i=0
    while(i<30):
        if(len(temp_input)>100):
            x_input=np.array(temp_input[1:])
            print("{} day input {}".format(i,x_input))
            x_input=x_input.reshape(1,-1)
            x_input = x_input.reshape((1, n_steps, 1))
            yhat = Model.predict(x_input, verbose=0)
            print("{} day output {}".format(i,yhat))
            temp_input.extend(yhat[0].tolist())
            temp_input=temp_input[1:]
            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            x_input = x_input.reshape((1, n_steps,1))
            yhat = Model.predict(x_input, verbose=0)
            print(yhat[0])
            temp_input.extend(yhat[0].tolist())
            print(len(temp_input))
            lst_output.extend(yhat.tolist())
            i=i+1
    print(lst_output)
    day_new=np.arange(1,101)
    day_pred=np.arange(101,131)
    len(adf1)
    plt.plot(day_new,scaler.inverse_transform(adf1[1158:]))
    plt.plot(day_pred,scaler.inverse_transform(lst_output))

else:
    fdf=pd.read_csv('FB.csv')
    fdf1=fdf.reset_index()['close']
    scaler=MinMaxScaler(feature_range=(0,1))
    adf1=scaler.fit_transform(np.array(fdf1).reshape(-1,1))
    training_size=int(len(fdf1)*0.65)
    test_size=len(fdf1)-training_size
    train_data,test_data=fdf1[0:training_size,:],fdf1[training_size:len(fdf1),:1]
    def create_dataset(dataset,time_step=1):
        dataX,dataY=[],[]
        for i in range(len(dataset)-time_step-1):
            a=dataset[i:(i+time_step), 0]
            dataX.append(a)
            dataY.append(dataset[i+time_step,0])
        return np.array(dataX),np.array(dataY)
    time_step=100
    X_train,y_train=create_dataset(train_data,time_step)
    X_test,y_test=create_dataset(test_data,time_step)
    X_train=X_train.reshape(X_train.shape[0],X_train.shape[1],1)
    X_test=X_test.reshape(X_test.shape[0],X_test.shape[1],1)
    train_predict=Model.predict(X_train)
    test_predict=Model.predict(X_test)
    train_predict=scaler.inverse_transform(train_predict)
    test_predict=scaler.inverse_transform(test_predict)
    look_back=100
    trainPredictPlot=np.empty_like(fdf1)
    trainPredictPlot[:, :]=np.nan
    trainPredictPlot[look_back:len(train_predict)+look_back, :]=train_predict
    testPredictPlot=np.empty_like(fdf1)
    testPredictPlot[:, :]=np.nan
    testPredictPlot[len(train_predict)+(look_back*2)+1:len(fdf1)-1:]=test_predict
    plt.plot(scaler.inverse_transform(fdf1))
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.show()
    x_input=test_data[341:].reshape(1,-1)
    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    lst_output=[]
    n_steps=100
    i=0
    while(i<30):
        if(len(temp_input)>100):
            x_input=np.array(temp_input[1:])
            print("{} day input {}".format(i,x_input))
            x_input=x_input.reshape(1,-1)
            x_input = x_input.reshape((1, n_steps, 1))
            yhat = Model.predict(x_input, verbose=0)
            print("{} day output {}".format(i,yhat))
            temp_input.extend(yhat[0].tolist())
            temp_input=temp_input[1:]
            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            x_input = x_input.reshape((1, n_steps,1))
            yhat = Model.predict(x_input, verbose=0)
            print(yhat[0])
            temp_input.extend(yhat[0].tolist())
            print(len(temp_input))
            lst_output.extend(yhat.tolist())
            i=i+1
    print(lst_output)
    day_new=np.arange(1,101)
    day_pred=np.arange(101,131)
    len(fdf1)
    plt.plot(day_new,scaler.inverse_transform(fdf1[1158:]))
    plt.plot(day_pred,scaler.inverse_transform(lst_output))
