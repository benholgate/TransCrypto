#!/usr/bin/env python
# coding: utf-8

# # Bitcoin BERT Train Dataset

# ## Dataset to experiment with BERT Transformer Models

# ## Using reduced dataset for computational time (4 Feb only)

# In[1]:


# Import libraries
from datetime import datetime
import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import nltk
import itertools
import collections
import csv
from wordcloud import WordCloud


# In[2]:


# Load Twitter tweets
lines = []
with open('BTCtweets_4Feb2021_EXP.txt') as f:
    lines = f.readlines()

# Check no. of tweets in dataset
print(len(lines))


# In[3]:


# Separate tweet text from date on each line
new_lines = []
for i in range(len(lines)):
  newline = lines[i].split('""')
  new_lines.append(newline)

print(len(new_lines))


# In[4]:


# Convert list into pandas dataframe
df = pd.DataFrame(new_lines, columns=['Tweet', 'Date'])
df.head()


# In[5]:


df.tail()


# In[6]:


# Drop first row
df = df.iloc[1:]
df.head()


# In[7]:


# Clean the tweets
def cleanTwt(twt):
    twt = re.sub("#bitcoin", 'bitcoin', twt) # remove the '#' from bitcoin
    twt = re.sub("#Bitcoin", 'Bitcoin', twt) # remove the '#' from Bitcoin
    twt = re.sub('#[A-Za-z0-9]+', '', twt) # remove any string with a '#'
    twt = re.sub('\\n', '', twt) # remove the '\n' string
    twt = re.sub('https:\/\/\S+', '', twt) # remove any hyperlinks

    twt = re.sub('(RT|via)((?:\\b\\W*@\\w+)+)', ' ', twt)
    twt = re.sub(r'@\S+', '', twt)
    twt = re.sub('&amp', ' ', twt)

    twt = re.sub(r"\s{2,}", " ", twt)  # remove multiple whitespaces
    twt = re.sub(r"\s+t\s+", "'t ", twt)  # replace separately standing "t" as 't
    twt = re.sub(r"”", "", twt)
    twt = re.sub(r"“", "", twt)
    #twt = re.sub(r"http stks co \w+\s*", "", twt) --- already done above???
    twt = re.sub(r"\w+tag_", "", twt)
    twt = re.sub(r"\w+tag\s*", "", twt)
    # remove punctuation and whitespaces from both ends
    #translator = str.maketrans('', '', string.punctuation) -- CANNOT ACTIVATE BEFORE VADER!!!
    #twt = twt.translate(translator).strip()
    # convert words to lower case and split them
    # twt = twt.lower().split(" ") --- CANNOT ACTIVATE BEFORE TEXTBLOB SENTIMENT!!!

    return twt

df['Clean_Tweet'] = df['Tweet'].apply(cleanTwt)
df.head()


# ## VADER Sentiment Analysis

# In[8]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')


# In[9]:


analyzer = SentimentIntensityAnalyzer()

df['neg'] = df['Clean_Tweet'].apply(lambda x:analyzer.polarity_scores(x)['neg'])
df['neu'] = df['Clean_Tweet'].apply(lambda x:analyzer.polarity_scores(x)['neu'])
df['pos'] = df['Clean_Tweet'].apply(lambda x:analyzer.polarity_scores(x)['pos'])
df['compound'] = df['Clean_Tweet'].apply(lambda x:analyzer.polarity_scores(x)['compound'])
df.head()


# In[10]:


# Create sentiment score for each tweet
def getCompoundSentiment(twt):
    cs_score = analyzer.polarity_scores(twt)['compound']
    if cs_score < 0:
        return "negative"
    elif cs_score > 0:
        return "positive"
    elif cs_score == 0:
        return "neutral"

df['Sentiment'] = df['Clean_Tweet'].apply(getCompoundSentiment)
df.head()


# In[11]:


df['compound'].describe()


# In[12]:


df['compound'].hist(bins=20)


# In[13]:


# Create bar chart to show the count of Positive, Neutral and Negative sentiments
df['Sentiment'].value_counts().plot(kind="bar")
plt.title("Sentiment Analysis Bar Graph")
plt.xlabel("Sentiment")
plt.ylabel("No. of tweets")
plt.show()


# In[14]:


df['Sentiment'].value_counts()


# In[15]:


# Drop all rows where sentiment = neutral
df2 = df.drop(df[df.Sentiment == 'neutral'].index)


# In[16]:


df2.shape


# In[17]:


df2['compound'].hist(bins=20)


# In[18]:


# Create bar chart to show the count of Positive and Negative sentiments
df2['Sentiment'].value_counts().plot(kind="bar")
plt.title("Sentiment Analysis Bar Graph")
plt.xlabel("Sentiment")
plt.ylabel("No. of tweets")
plt.show()


# In[19]:


df2['Sentiment'].value_counts()


# In[20]:


# Reduce no. of columns to Clean_Tweet and Sentiment only
df3 = df2[['Clean_Tweet', 'Sentiment']]
df3.head()


# In[21]:


# Change Sentiment to 0 for negative, 1 for positive
df3['Sentiment'] = df3['Sentiment'].replace(['positive','negative'], ['1','0'])
df3.head()


# In[22]:


df3.shape


# In[23]:


# Save dataframe as a CSV file
df3_csv_data = df3.to_csv('BTC_BERT_Train_Dataset.csv', index = False)
print('\nBTC_BERT_Train_Dataset:\n', df3_csv_data)


# In[26]:


# Split dataset into positive and negative
df_pos = df3[df3['Sentiment'] == '1']
df_neg = df3[df3['Sentiment'] == '0']


# In[27]:


df_pos.shape


# In[28]:


df_neg.shape


# In[29]:


''' Select from df_pos same number of rows
as in df_neg at randonm so that both datasets
are the same length. '''
df_pos = df_pos.sample(n=9154)
df_pos.shape


# In[30]:


# Merge df_pos and df_neg into one dataframe
df4 = df_pos.append(df_neg)
df4.head()


# In[31]:


df4.tail()


# In[32]:


df4.shape


# In[33]:


# Save dataframe as a CSV file
df4_csv_data = df4.to_csv('BTC_BERT_Train_Dataset_Balanced.csv', index = False)
print('\nBTC_BERT_Train_Dataset:\n', df4_csv_data)

