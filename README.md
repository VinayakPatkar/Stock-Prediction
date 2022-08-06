# Stock-Prediction
The system is built for predicting stocks using LSTM. A typical second year project built with literally no knowledge about deep learning.

The system uses LSTM  for predicting stock prices. (Special mentions for Krish Naik for info on LSTM)

The system analyses the data from the TIINGO API of the particular stocks mentioned in the system.

Streamlit was used for building the GUI for the system.

Sentimental analysis was done using the news and twitter data.

Further improvements can be achieved using various algorithms.

FMODEL.h5,Mfinalmodel.h5,ModelNew.h5,NewMod.h5 are the 4 models created using LSTM.

FMODEL.h5 gave the best accuracy for predicting stocks.

NewsAnalyzer and SentimentAnalyzer process the data and perform sentimental analysis on the data.

app2.py is the GUI file for the system.

To run the code:

Install all the necessary modules.

Run streamlit run app2.py in cmd

*Network Access required for running the code

*Not to be used in real life for stock trading
