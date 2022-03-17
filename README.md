# TransCrypto
## Objectives
The aim of this project was to develop a predictive model using a pre-trained Transformer neural network that can determine investor sentiment on the most traded cryptocurrency, Bitcoin, and determine whether the Bitcoin price is likely to move up or down as a result of that sentiment. The model name - TransCrypto - is a play on Transformer + cryptocurrency.

## Datasets
A cryptocurrency has no instrinsic value and its value is therefore mostly dictated by investor sentiment. I used Twitter tweets about Bitcoin as a proxy for investor sentiment, limiting those tweets to US origination, in the English language, and published on two specific months. I chose two of the most volatile months in Bitcoin’s price prior to the study: February 2021, when the price spiked, and May 2021, when the price crashed. The dataset consisted of 3.4 million tweets in February and 5.2 million tweets in May.

![image](https://user-images.githubusercontent.com/69303050/158633373-1828e60b-311f-4e1f-a735-fb6a5febb9e7.png)

## Results
The model found a strong correlation between daily investor sentiment (as reflected by Twitter tweets) and the Bitcoin price the following day, with a correlation coefficient of about 0.77 on those two months. In other words, if sentiment was positive, the price would likely rise the next day, conversely, if sentiment was negative, the price would likely fall.

## Model Architecture
The TransCrypto model is centred around the DistilBERT pre-trained Transformer model. I chose DistilBERT because it's 60% faster than the original pre-trained Transformer, BERT, but retains 97% of BERT's language understanding capabilities. The illustration below represents the TransCrypto model’s architecture and its data flow. There are two datasets: 1) tweets about Bitcoin downloaded from Twitter’s v2 API; and 2) Bitcoin’s historical prices downloaded from Yahoo Finance. A sample of Bitcoin tweets was labelled by NLTK's VADER sentiment algorithm to indicate whether each tweet in the sample is positive or negative (eliminating neutral tweets, which in theory should not influence price). This labelled sample dataset was used to fine-tune the DistilBERT model. Subsequentlty, the DistilBERT model predicted positive or negative sentiment in the remainder of unlabelled tweets. Finally, another program compared the labelled tweets against historical Bitcoin prices over the same time period using regression analysis. The programs for each of these steps are included in this repository. I used Python with TensorFlow and Keras.

![image](https://user-images.githubusercontent.com/69303050/158636151-4dfb1758-3f68-4255-9a21-5f6d4c68942e.png)

As an extra exercise, I found that the fine-tuned DistilBERT model can create positive and negative word lexicons specific to Bitcoin tweets that in turn can be used as the basis for a new dataset to fine-tune the model for subsequent iterations.

At the time, this was the first project to: 1) employ a pre-Trained neural network to examine the relationship between Twitter sentiment about Bitcoin and Bitcoin trading; 2) develop positive and negtive word lexicons tailored specifically for Bitcoin; and 3) develop a model that can create a truly dynamic process for word lexicons about Bitcoin (as distinct from typically static word lexicons for finance that remain unchanged after publication).
