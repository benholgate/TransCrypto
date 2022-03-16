# transcrypto
The aim of this project was to develop a prediction model using a pre-trained Transformer neural network that can determine investor sentiment on the most traded cryptocurrency, Bitcoin, and whether the Bitcoin price is likely to move up or down as a result of that sentiment. The model name - Transcrypto - is a play on Transformer + cryptocurrency.

A cryptocurrency has no instrinsic value and its value is therefore mostly dictated by investor sentiment. I used Twitter tweets about Bitcoin as a proxy for investor sentiment, limiting those tweets to US origination, in the English language, and published on two specific months. I chose two of the most volatile months in Bitcoin’s price prior to the study: February 2021, when the price spiked, and May 2021, when the price crashed. The dataset consisted of 3.4 million tweets in February and 5.2 million tweets in May.

![image](https://user-images.githubusercontent.com/69303050/158633373-1828e60b-311f-4e1f-a735-fb6a5febb9e7.png)

The model found a strong correlation between daily investor sentiment (as reflected by Twitter tweets) and the Bitcoin price the following day, with a correlation coefficient of about 0.77. In other words, if sentiment was positive, the price would likely rise the next day, conversely, if sentiment was negative, the price would likely fall.

![image](https://user-images.githubusercontent.com/69303050/158636151-4dfb1758-3f68-4255-9a21-5f6d4c68942e.png)

sdsds


