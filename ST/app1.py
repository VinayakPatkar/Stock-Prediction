import streamlit as stm
import pandas as pd
import pandas_datareader as pdr
import plotly.graph_objects as go
import numpy as np
import pandas_datareader as pdr
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow import keras
import plotly.express as px
import tensorflow as tf
dfk=np.arange(start=1,stop=1259)
stm.title('Stock Prediction System')
stocks=("AAPL","TSLA","MSFT")
selected_stock=stm.selectbox("Select stock:",stocks)
def load_data(ticker):
    df=pdr.get_data_tiingo(ticker,api_key='c546891fca3ed571aa1ae049569f17d5223b8e87')
    df.rename( columns={1 :'Date'}, inplace=True )
    return df
data_load_state=stm.text('Load Data....')
data=load_data(selected_stock)
data_load_state.text("Loading data...done!")
stm.subheader('Raw Data')
stm.write(data.tail())
def plot_raw_data():
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=dfk,y=data['close'],name='stock_open'))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    stm.plotly_chart(fig)

plot_raw_data()

data1=data.reset_index()['close']

scaler=MinMaxScaler(feature_range=(0,1))
data1=scaler.fit_transform(np.array(data1).reshape(-1,1))
training_size=int(len(data1)*0.65)
test_size=len(data1)-training_size
train_data,test_data=data1[0:training_size,:],data1[training_size:len(data1),:1]
def create_dataset(dataset,time_step=1):
    dataX,dataY=[],[]
    for i in range(len(dataset)-time_step-1):
        a=dataset[i:(i+time_step),0]
        dataX.append(a)
        dataY.append(dataset[i+time_step,0])
    return np.array(dataX), np.array(dataY)
time_step=100
X_train,y_train=create_dataset(train_data,time_step)
X_test,ytest=create_dataset(test_data,time_step)
X_train=X_train.reshape(X_train.shape[0],X_train.shape[1],1)
X_test=X_test.reshape(X_test.shape[0],X_test.shape[1],1)
model=keras.models.load_model('NewMod.h5')
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)
look_back=100
trainPredictPlot=np.empty_like(data1)
trainPredictPlot[:, :]=np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :]=train_predict
testPredictPlot=np.empty_like(data1)
testPredictPlot[:, :]=np.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(data1)-1, :]=test_predict
x_input=test_data[341:].reshape(1,-1)
temp_input=list(x_input)
temp_input=temp_input[0].tolist()
from numpy import array
lst_output=[]
n_steps=100
i=0
while(i<30):
    if(len(temp_input)>100):
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input=x_input.reshape((1,n_steps,1))
        yhat=model.predict(x_input,verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input=x_input.reshape((1,n_steps,1))
        yhat=model.predict(x_input,verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
print(lst_output)
day_new=np.arange(1,101)
day_pred=np.arange(101,131)
df3=data1.tolist()
df3.extend(lst_output)
day_new=scaler.inverse_transform(data1[1158:])
day_pred=scaler.inverse_transform(lst_output)

#plt.plot(scaler.inverse_transform(data1))
#plt.plot(trainPredictPlot)
#plt.plot(testPredictPlot)
#plt.show()