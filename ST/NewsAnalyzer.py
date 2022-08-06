import requests
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from textblob import TextBlob
from string import Template
from datetime import date,timedelta
today=date.today();
d1 = today.strftime("%Y-%m-%d")
end_date = d1
d2 = date.today() - timedelta(days=25)
d2 = d2.strftime("%Y-%m-%d")
start_date = d2
print(end_date);
print(start_date)

def NewsAnalyzer(ticker):
    url = ('https://newsapi.org/v2/everything?'
            'q='+ticker+'&'
            'from='+start_date+'&'
            'sortBy=popularity&'
            'apiKey=5de90078498543aaa7bd0c613083233e')
    print(url)
    page = requests.get(url).json() 
    print(page)
    article = page["articles"] 
    #print(article)
    main_titles=[]
    for art in article:
        main_titles.append(art['title'])
    def clean(text):
        text = re.sub('@[A-Za-z0–9]+', '', text) 
        text = re.sub('#', '', text) 
        text = re.sub('RT[\s]+', '', text) 
        text = re.sub('https?:\/\/\S+', '', text) 
        return text
    for title in main_titles:
        title=clean(title)
    def percentage(part,whole):
        return 100 * float(part)/float(whole)

    #Assigning Initial Values
    positive = 0
    negative = 0
    neutral = 0
    #Creating empty lists
    articles1 = []
    neutral_list = []
    negative_list = []
    positive_list = []

    #Iterating over the tweets in the dataframe
    for title in main_titles:
        articles1.append(title)
        analyzer = SentimentIntensityAnalyzer().polarity_scores(title)
        print(title)
        neg = analyzer['neg']
        print(neg)
        neu = analyzer['neu']
        print(neu)
        pos = analyzer['pos']
        print(pos)
        comp = analyzer['compound']
        print(comp)

        if neg > pos:
            negative_list.append(title) 
            negative += 1 
        elif pos > neg:
            positive_list.append(title)
            positive += 1 
        elif pos == neg:
            neutral_list.append(title) 
            neutral += 1 

    positive = percentage(positive, len(main_titles)) 
    negative = percentage(negative, len(main_titles))
    neutral = percentage(neutral, len(main_titles))
    print('Based on vaders performance');
    print('Positive:',positive);
    print('Neutral:',neutral);
    print('Negative:',negative);

    gp=0;
    for title in main_titles:
        blob= TextBlob(title)
        p = 0
        for sentence in blob.sentences:
            p += sentence.sentiment.polarity
            gp += sentence.sentiment.polarity
    gp=gp/len(main_titles);
    print('Using TextBlob')
    print(gp)
    return positive,neutral,negative,gp

gp=NewsAnalyzer('AAPL')
print(gp)