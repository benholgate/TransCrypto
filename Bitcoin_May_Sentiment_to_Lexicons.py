#!/usr/bin/env python
# coding: utf-8

# # Convert Bitcoin May Sentiment Compile into Pos / Neg Lexicons

# ## Uses 620,000 May tweets that Transformer NN

# ## applied pos / neg labels (model trained on VADER tweets)

# In[47]:


# Import libraries
import pandas as pd
import numpy as np
import re
import csv
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import itertools
import collections
import string


# In[11]:


# reading the CSV file
df1 = pd.read_csv('BTC_May_Sentiment_Datasets_Compile.csv')
df1.head()


# In[12]:


df1.shape


# In[13]:


# Reduce no. of columns to Clean_Tweet and Model Sentiment only
df2 = df1[['Clean_Tweet', 'model_sentiment']]
df2.head()


# In[18]:


df2['model_sentiment'].value_counts()


# In[20]:


# Create bar chart to show the count of Positive, Neutral and Negative sentiments
df2['model_sentiment'].value_counts().plot(kind="bar")
plt.title("Sentiment Analysis Bar Graph")
plt.xlabel("Sentiment")
plt.ylabel("No. of tweets")
plt.show()


# In[30]:


# Split dataset into positive and negative datasets
df_pos = df2[df2['model_sentiment'] == 'Positive']
df_neg = df2[df2['model_sentiment'] == 'Negative']


# In[31]:


df_pos.shape


# In[32]:


df_neg.shape


# In[15]:


nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


# In[16]:


newStopWords = ['...', ' ... ', '-', ':', '1', '|', 
                '.', ',', "’", ':', ';', '\\', '//', '#',
                '*', '(', ')', '<', '>', '~', '^', "'",
                '{', '}', '[', ']', '¿', '|', '"', "&",
                '/', '_', '`', '', '\t', '\n',
                'quot', 'http', 'https', 'com',
                'www', 'an', 'the']
stop_words.update(newStopWords)


# ### Analyse Positive Tweet Dataset

# In[33]:


# Create word cloud
text_wc = df_pos['Clean_Tweet'].to_string().lower()
wordcloud = WordCloud(
    collocations=False,
    background_color='white',
    relative_scaling=0.5,
    stopwords=stop_words).generate(text_wc)

plt.figure(figsize=(14,14))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()


# Some of the code below adapted from:
# https://www.earthdatascience.org/courses/use-data-open-source-python/intro-to-apis/calculate-tweet-word-frequencies-in-python/

# In[34]:


# Reduce positive tweets dataset to text only
df_pos = df_pos['Clean_Tweet']


# In[35]:


# Create a list of lists containing lowercase words for each tweet
words_in_tweet_pos = [tweet.lower().split() for tweet in df_pos]


# In[36]:


# Remove stop words from each tweet list of words
words_in_tweet_pos = [[word for word in tweet_words if not word in stop_words]
              for tweet_words in words_in_tweet_pos]


# In[48]:


# List of all words across tweets
all_words_pos = list(itertools.chain(*words_in_tweet_pos))

# Create counter
counts = collections.Counter(all_words_pos)

# Create lists of most common words
df_most_common_pos_30 = pd.DataFrame(counts.most_common(30),
                             columns=['words', 'count'])

df_most_common_pos_500 = pd.DataFrame(counts.most_common(500),
                             columns=['words', 'count'])


# In[63]:


# Create bar chart of 30 most common words in positive tweets 
ax = df_most_common_pos_30.sort_values(by='count').plot.barh(x='words', y='count', figsize=(8, 8))
plt.title('Most Common Words in Positive Tweets')
plt.ylabel('Words')
plt.xlabel('Count')
plt.show()


# In[64]:


# Save df_most_common_pos_500 as a CSV file
Bitcoin_most_common_pos_500_from_May = df_most_common_pos_500.to_csv('Bitcoin_most_common_pos_500_from_May.csv', index = False)
print('\nBitcoin_May_Sentiment_to_Lexicons:\n', Bitcoin_most_common_pos_500_from_May)


# ### Analyse Negative Tweet Dataset

# In[66]:


# Create word cloud
text_wc = df_neg['Clean_Tweet'].to_string().lower()
wordcloud = WordCloud(
    collocations=False,
    background_color='white',
    relative_scaling=0.5,
    stopwords=stop_words).generate(text_wc)

plt.figure(figsize=(14,14))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()


# In[67]:


# Reduce negative tweets dataset to text only
df_neg = df_neg['Clean_Tweet']


# In[68]:


# Create a list of lists containing lowercase words for each tweet
words_in_tweet_neg = [tweet.lower().split() for tweet in df_neg]


# In[69]:


# Remove stop words from each tweet list of words
words_in_tweet_neg = [[word for word in tweet_words if not word in stop_words]
              for tweet_words in words_in_tweet_neg]


# In[70]:


# List of all words across tweets
all_words_neg = list(itertools.chain(*words_in_tweet_neg))

# Create counter
counts = collections.Counter(all_words_neg)

# Create lists of most common words
df_most_common_neg_30 = pd.DataFrame(counts.most_common(30),
                             columns=['words', 'count'])

df_most_common_neg_500 = pd.DataFrame(counts.most_common(500),
                             columns=['words', 'count'])


# In[71]:


# Create bar chart of 30 most common words in negative tweets 
ax = df_most_common_neg_30.sort_values(by='count').plot.barh(x='words', y='count', figsize=(8, 8))
plt.title('Most Common Words in Positive Tweets')
plt.ylabel('Words')
plt.xlabel('Count')
plt.show()


# In[72]:


# Save df_most_common_neg_500 as a CSV file
Bitcoin_most_common_neg_500_from_May = df_most_common_neg_500.to_csv('Bitcoin_most_common_neg_500_from_May.csv', index = False)
print('\nBitcoin_May_Sentiment_to_Lexicons:\n', Bitcoin_most_common_neg_500_from_May)


# Nasekin and Chen (2020) develop their own finance lexicon of 2,597 words by using a seed dictionary of 1,311 finance terms by Renault (2017) and constructing an additional 1,286 finance terms based on their analysis of Stocktwits forums. Each word in their index has a sentiment score in the interval of [-1, 1]. 

# In[ ]:




