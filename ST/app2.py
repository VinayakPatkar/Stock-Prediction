#Importing the modules
import streamlit as st
import datetime as dt
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
scaler=MinMaxScaler(feature_range=(0,1))
from NewsAnalyzer import *
from SentimentAnalyzer import *


#API-KEY
key="c546891fca3ed571aa1ae049569f17d5223b8e87"

#Headers and title
st.title('Stock Prediction System')
st.text("ATTENTION:Investing in stocks is associated with financial risks.Please invest carefully")

#Dropdown for stock selection
stocks=("AAPL","TSLA","MSFT","AMZN","GOOG")
selected_stock=st.selectbox("Select stock:",stocks)


#Function for accessing data from the api and keeping the closed dataset only
def load_data(ticker):
    data = pdr.get_data_tiingo(ticker, api_key=key)
    df1=data.reset_index()['close']
    return df1;

#Highlighters to signal that data is accessed
data_load_state=st.text('Load Data....')
data=load_data(selected_stock)
data_load_state.text("Loading data...done!")

length=len(data)
dfk=np.arange(start=1,stop=length)

#Function to close plot data 
def plot_raw_data():
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=dfk,y=data,name='stock_open'))
    fig.layout.update(title_text="Close Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
plot_raw_data()


#Importing the model
model1=keras.models.load_model('FMODEL.h5')

#Signal that preprocessing has started
data_preprocessing_state=st.text('Preprocessing the data')

#Terms included in Preprocessing
data_preprocessing_terms=st.text('Data Processing includes Scaling,Dataset Dividing and shaping the data')
#Creating the Time Series Dataset
def create_dataset(dataset,time_step=1):
    dataX,dataY=[],[]
    for i in range(len(dataset)-time_step-1):
        a=dataset[i:(i+time_step),0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

def gettingTestData(dataset):
    dataset=scaler.fit_transform(np.array(dataset).reshape(-1,1))
    training_size=int(len(dataset)*0.65)
    test_size=len(dataset)-training_size
    train_data,test_data=dataset[0:training_size,:],dataset[training_size:len(dataset),:1]
    return train_data,test_data
train_data,test_data=gettingTestData(data)

#Preprocessing the data
def Preprocessing(dataset):
    dataset=scaler.fit_transform(np.array(dataset).reshape(-1,1))
    training_size=int(len(dataset)*0.65)
    test_size=len(dataset)-training_size
    train_data,test_data=dataset[0:training_size,:],dataset[training_size:len(dataset),:1]
    time_step = 100
    X_train, y_train = create_dataset(train_data, time_step)  
    X_test, ytest = create_dataset(test_data, time_step)
    X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)
    return X_test,ytest,

#Obtaining the training and testing dataset
X_test,ytest=Preprocessing(data)

#Signal that preprocessing is done
data_preprocessing_state_loaded=st.text('Preprocessing Done')

st.info('Note: The model is created using LSTM')
st.info('Model Statistics')
dataframe={'Mean Squared Error':[216.91764384115888],'Mean Absolute Error':[146.1769179267511],'Mean Absolute Percentage Error':[2003.6777762292625]}
new_dataframe=pd.DataFrame(dataframe)
st.table(new_dataframe)
x_input=test_data[341:].reshape(1,-1)
temp_input=list(x_input)
temp_input=temp_input[0].tolist()
from numpy import array

lst_output=[]
n_steps=100
i=0
while(i<8):
    
    if(len(temp_input)>100):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
        yhat = model1.predict(x_input, verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model1.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
    
print(lst_output)
new_output=[]
new_output=scaler.inverse_transform(lst_output)
print(new_output)
st.info('Prediction for next 7 days from today')
dataframe1={'Day1':new_output[0],'Day2':new_output[1],'Day3':new_output[2],'Day4':new_output[3],'Day5':new_output[4],'Day6':new_output[5],'Day7':new_output[6]}
new_dataframe1=pd.DataFrame(dataframe1)
st.table(new_dataframe1)
positive,neutral,negative,gp=NewsAnalyzer(selected_stock);
st.title('Sentimental Analysis Indicators:')
st.info('Various NLP techniques were used for sentimental analysis');
st.info('Sentiment analysis fluctuates. Only to be used as an indicator')
st.header('News Analysis')
st.subheader('Using Vader analysis');
Values=[];
Values.append(positive)
Values.append(neutral)
Values.append(negative)
dataframe2={'Positive':[Values[0]],'Neutral':[Values[1]],'Negative':[Values[2]]}
new_dataframe2=pd.DataFrame(dataframe2);
st.table(new_dataframe2)
st.subheader('Using TextBlob');
if(gp>0):
    st.text('News analysis provides positive sentiment')
    #st.text('Sentiment Value:')
    st.text(gp);
elif(gp<0):
    st.text('News analysis provides negative sentiment')
    #st.text('Sentiment Value:')
    st.text(gp);
else:
    st.text('News analysis provides neutral sentiment')
    #st.text('Sentiment Value:')
    st.text(gp);

print(gp)
st.header('Twitter Analysis')
st.subheader('Using Vader analysis');
positiven,neutraln,negativen,gpn=TweetAnalyzer(selected_stock);
print(positiven,neutraln,negativen,gpn)
dataframe3={'Positive':[positiven],'Neutral':[neutraln],'Negative':[negativen]}
new_dataframe3=pd.DataFrame(dataframe3);
st.table(new_dataframe3)
st.subheader('Using TextBlob');
if(gpn>0):
    st.text('Twitter analysis provides positive sentiment')
    #st.text('Sentiment Value:')
    st.text(gpn);
elif(gpn<0):
    st.text('Twitter analysis provides negative sentiment')
    #st.text('Sentiment Value:')
    st.text(gpn);
else:
    st.text('Twitter analysis provides neutral sentiment')
    #st.text('Sentiment Value:')
    st.text(gpn);