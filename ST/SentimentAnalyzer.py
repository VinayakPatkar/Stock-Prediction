#For visualizing and cleaning the data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import re
from wordcloud import WordCloud, STOPWORDS
#Sentiment Analyzer 
from nltk.sentiment.vader import SentimentIntensityAnalyzer
#For the tweets
import snscrape.modules.twitter as sntwitter
import nltk
from textblob import TextBlob
#Companies to be searched for (More modification required)
def TweetAnalyzer(ticker):
    #"AAPL","TSLA","MSFT","AMZN","GOOG"
    if(ticker=='AAPL'):
        Company='Apple';
    elif(ticker=='TSLA'):
        Company='Tesla';
    elif(ticker=='MSFT'):
        Company='Microsoft';
    elif(ticker=='AMZN'):
        Company='Amazon';
    elif(ticker=='GOOG'):
        Company='Google';
    if Company != '':
        #Can be altered according to the choice if required
        noOfTweet = 1000
        noOfDays = 10
        #List to store the tweets
        nlist = []
        text_list=[]
        #Todays Date
        now = dt.date.today()
        #Converting to string
        now = now.strftime('%Y-%m-%d')
        print(now)
        #TimeDelta for going back noOfDays
        yesterday = dt.date.today() - dt.timedelta(days = int(noOfDays))
        yesterday = yesterday.strftime('%Y-%m-%d')
        print(yesterday)
        #Enumerate for the showing the counter as well as data and the time llimit from the day specified till today
        for i,tweet in enumerate(sntwitter.TwitterSearchScraper(Company + ' lang:en since:' +  yesterday + ' until:' + now + ' -filter:links -filter:replies').get_items()):
            if i > int(noOfTweet):
                    break
            nlist.append([tweet.date, tweet.id, tweet.content, tweet.username])
            text_list.append(tweet.content)
        #Creating a dataframe
        df = pd.DataFrame(nlist, columns=['Datetime', 'Tweet Id', 'Text', 'Username'])
        print(df)
    print(text_list)
    #Cleaning the downloaded data
    def clean(text):
        text = re.sub('@[A-Za-z0–9]+', '', text) 
        text = re.sub('#', '', text) 
        text = re.sub('RT[\s]+', '', text) 
        text = re.sub('https?:\/\/\S+', '', text) 
        return text

    #applying this function to Text column of our dataframe
    df["Text"] = df["Text"].apply(clean)
    print(df["Text"][0])

    def percentage(part,whole):
        return 100 * float(part)/float(whole)

    #Assigning Initial Values
    positive = 0
    negative = 0
    neutral = 0
    #Creating empty lists
    tweet_list1 = []
    neutral_list = []
    negative_list = []
    positive_list = []

    #Iterating over the tweets in the dataframe
    for tweet in df['Text']:
        tweet_list1.append(tweet)
        analyzer = SentimentIntensityAnalyzer().polarity_scores(tweet)
        print(tweet)
        neg = analyzer['neg']
        print(neg)
        neu = analyzer['neu']
        print(neu)
        pos = analyzer['pos']
        print(pos)
        comp = analyzer['compound']
        print(comp)

        if neg > pos:
            negative_list.append(tweet) 
            negative += 1 
        elif pos > neg:
            positive_list.append(tweet)
            positive += 1 
        elif pos == neg:
            neutral_list.append(tweet) 
            neutral += 1 

    positive = percentage(positive, len(df)) 
    negative = percentage(negative, len(df))
    neutral = percentage(neutral, len(df))
    print('Based on vaders performance');
    print('Positive:',positive);
    print('Neutral:',neutral);
    print('Negative:',negative);

    gp=0;
    for tweet in text_list:
        blob= TextBlob(tweet)
        p = 0
        for sentence in blob.sentences:
            p += sentence.sentiment.polarity
            gp += sentence.sentiment.polarity
    gp=gp/len(text_list);
    print('Using TextBlob')
    print(gp)
    return positive,neutral,negative,gp;
TweetAnalyzer('AAPL')