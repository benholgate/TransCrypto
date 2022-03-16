#!/usr/bin/env python
# coding: utf-8

# ## Twitter Sentiment & Bitcoin Regression Analysis

# Code adapted from: https://datatofish.com/multiple-linear-regression-python/

# In[1]:


# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import statsmodels.api as sm


# ## Neg / Pos Ratio & Bitcoin Price

# ### May 2021 Dataset

# In[2]:


# Download May 2021 data
df_May = pd.read_csv('Regression_Ratios_BTCprice_May.csv')
df_May.head()


# #### VADER Analysis: 1-31 May

# In[3]:


# Plot VADER vs Bitcoin price
plt.scatter(df_May['VADER Neg / Pos Ratio'], df_May['Bitcoin Price (Adj Close)'], color='green')
plt.title('VADER Neg/Pos Ratio vs Bitcoin Price: 1-31 May', fontsize=14)
plt.xlabel('VADER Neg/Pos Ratio', fontsize=14)
plt.ylabel('Bitcoin Price', fontsize=14)
plt.grid(True)
plt.show()


# In[4]:


# Calculate correlation VADER vs Bitcoin price
df_May_Vcor = df_May[['VADER Neg / Pos Ratio', 'Bitcoin Price (Adj Close)']]
df_May_Vcor.corr()


# In[5]:


# Regression analysis: VADER vs Bitcoin price
X = df_May['VADER Neg / Pos Ratio']
Y = df_May['Bitcoin Price (Adj Close)']
# with statsmodels
X = sm.add_constant(X) # adding a constant
 
model = sm.OLS(Y, X).fit()
predictions = model.predict(X) 
 
print_model = model.summary()
print(print_model)


# #### VADER Analysis: 1-22 May

# In[6]:


# Reduce dataset to 1-22 May
df_May_reduced = df_May[0:22]


# In[7]:


# Plot VADER vs Bitcoin price
plt.scatter(df_May_reduced['VADER Neg / Pos Ratio'], df_May_reduced['Bitcoin Price (Adj Close)'], color='green')
plt.title('VADER Neg/Pos Ratio vs Bitcoin Price: 1-22 May', fontsize=14)
plt.xlabel('VADER Neg/Pos Ratio', fontsize=14)
plt.ylabel('Bitcoin Price', fontsize=14)
plt.grid(True)
plt.show()


# In[8]:


# Calculate correlation VADER vs Bitcoin price
df_May_reduced_Vcor = df_May_reduced[['VADER Neg / Pos Ratio', 'Bitcoin Price (Adj Close)']]
df_May_reduced_Vcor.corr()


# In[9]:


# Regression analysis: VADER vs Bitcoin price
X = df_May_reduced['VADER Neg / Pos Ratio']
Y = df_May_reduced['Bitcoin Price (Adj Close)']
# with statsmodels
X = sm.add_constant(X) # adding a constant
 
model = sm.OLS(Y, X).fit()
predictions = model.predict(X) 
 
print_model = model.summary()
print(print_model)


# #### Chen Analysis: 1-31 May

# In[10]:


# Plot Chen vs Bitcoin price
plt.scatter(df_May['Chen Neg / Pos Ratio'], df_May['Bitcoin Price (Adj Close)'], color='green')
plt.title('Chen Neg/Pos Ratio vs Bitcoin Price: 1-31 May', fontsize=14)
plt.xlabel('Chen Neg/Pos Ratio', fontsize=14)
plt.ylabel('Bitcoin Price', fontsize=14)
plt.grid(True)
plt.show()


# In[11]:


# Calculate correlation Chen vs Bitcoin price
df_May_Ccor = df_May[['Chen Neg / Pos Ratio', 'Bitcoin Price (Adj Close)']]
df_May_Ccor.corr()


# In[12]:


# Regression analysis: Chen vs Bitcoin price
X = df_May['Chen Neg / Pos Ratio']
Y = df_May['Bitcoin Price (Adj Close)']

# with statsmodels
X = sm.add_constant(X) # adding a constant
 
model = sm.OLS(Y, X).fit()
predictions = model.predict(X) 
 
print_model = model.summary()
print(print_model)


# #### Chen Analysis: 1-22 May

# In[13]:


# Plot Chen vs Bitcoin price
plt.scatter(df_May_reduced['Chen Neg / Pos Ratio'], df_May_reduced['Bitcoin Price (Adj Close)'], color='green')
plt.title('Chen Neg/Pos Ratio vs Bitcoin Price: 1-22 May 2021', fontsize=14)
plt.xlabel('Chen Neg/Pos Ratio', fontsize=14)
plt.ylabel('Bitcoin Price', fontsize=14)
plt.grid(True)
plt.show()


# In[14]:


# Calculate correlation Chen vs Bitcoin price
df_May_reduced_Ccor = df_May_reduced[['Chen Neg / Pos Ratio', 'Bitcoin Price (Adj Close)']]
df_May_reduced_Ccor.corr()


# In[15]:


# Regression analysis: Chen vs Bitcoin price
X = df_May_reduced['Chen Neg / Pos Ratio']
Y = df_May_reduced['Bitcoin Price (Adj Close)']
# with statsmodels
X = sm.add_constant(X) # adding a constant
 
model = sm.OLS(Y, X).fit()
predictions = model.predict(X) 
 
print_model = model.summary()
print(print_model)


# ### Feb 2021 Dataset

# In[16]:


# Download Feb 2021 data
df_Feb = pd.read_csv('Regression_Ratios_BTCprice_Feb.csv')
df_Feb.head()


# #### VADER Analysis: 1-28 Feb

# In[17]:


# Plot VADER vs Bitcoin price
plt.scatter(df_Feb['VADER Neg / Pos Ratio'], df_Feb['Bitcoin Price (Adj Close)'], color='green')
plt.title('VADER Neg/Pos Ratio vs Bitcoin Price: 1-28 Feb', fontsize=14)
plt.xlabel('VADER Neg/Pos Ratio', fontsize=14)
plt.ylabel('Bitcoin Price', fontsize=14)
plt.grid(True)
plt.show()


# In[18]:


# Calculate correlation VADER vs Bitcoin price
df_Feb_Vcor = df_Feb[['VADER Neg / Pos Ratio', 'Bitcoin Price (Adj Close)']]
df_Feb_Vcor.corr()


# In[19]:


# Regression analysis: VADER vs Bitcoin price
X = df_Feb['VADER Neg / Pos Ratio']
Y = df_Feb['Bitcoin Price (Adj Close)']
# with statsmodels
X = sm.add_constant(X) # adding a constant
 
model = sm.OLS(Y, X).fit()
predictions = model.predict(X) 
 
print_model = model.summary()
print(print_model)


# #### VADER Analysis: 1-20 Feb

# In[20]:


# Reduce dataset to 1-20 Feb
df_Feb_reduced = df_Feb[0:20]


# In[21]:


# Plot VADER vs Bitcoin price
plt.scatter(df_Feb_reduced['VADER Neg / Pos Ratio'], df_Feb_reduced['Bitcoin Price (Adj Close)'], color='green')
plt.title('VADER Neg/Pos Ratio vs Bitcoin Price: 1-20 Feb', fontsize=14)
plt.xlabel('VADER Neg/Pos Ratio', fontsize=14)
plt.ylabel('Bitcoin Price', fontsize=14)
plt.grid(True)
plt.show()


# In[22]:


# Calculate correlation VADER vs Bitcoin price
df_Feb_reduced_Vcor = df_Feb_reduced[['VADER Neg / Pos Ratio', 'Bitcoin Price (Adj Close)']]
df_Feb_reduced_Vcor.corr()


# In[23]:


# Regression analysis: VADER vs Bitcoin price
X = df_Feb_reduced['VADER Neg / Pos Ratio']
Y = df_Feb_reduced['Bitcoin Price (Adj Close)']
# with statsmodels
X = sm.add_constant(X) # adding a constant
 
model = sm.OLS(Y, X).fit()
predictions = model.predict(X) 
 
print_model = model.summary()
print(print_model)


# #### Chen Analysis: 1-28 Feb

# In[24]:


# Plot Chen vs Bitcoin price
plt.scatter(df_May['Chen Neg / Pos Ratio'], df_May['Bitcoin Price (Adj Close)'], color='green')
plt.title('Chen Neg/Pos Ratio vs Bitcoin Price: 1-28 Feb', fontsize=14)
plt.xlabel('Chen Neg/Pos Ratio', fontsize=14)
plt.ylabel('Bitcoin Price', fontsize=14)
plt.grid(True)
plt.show()


# In[25]:


# Calculate correlation Chen vs Bitcoin price
df_Feb_Ccor = df_Feb[['Chen Neg / Pos Ratio', 'Bitcoin Price (Adj Close)']]
df_Feb_Ccor.corr()


# In[26]:


# Regression analysis: Chen vs Bitcoin price
X = df_Feb['Chen Neg / Pos Ratio']
Y = df_Feb['Bitcoin Price (Adj Close)']
# with statsmodels
X = sm.add_constant(X) # adding a constant
 
model = sm.OLS(Y, X).fit()
predictions = model.predict(X) 
 
print_model = model.summary()
print(print_model)


# #### Chen Analysis: 1-20 Feb

# In[27]:


# Plot Chen vs Bitcoin price
plt.scatter(df_Feb_reduced['Chen Neg / Pos Ratio'], df_Feb_reduced['Bitcoin Price (Adj Close)'], color='green')
plt.title('Chen Neg/Pos Ratio vs Bitcoin Price: 1-20 Feb', fontsize=14)
plt.xlabel('Chen Neg/Pos Ratio', fontsize=14)
plt.ylabel('Bitcoin Price', fontsize=14)
plt.grid(True)
plt.show()


# In[28]:


# Calculate correlation Chen vs Bitcoin price
df_Feb_reduced_Ccor = df_Feb_reduced[['Chen Neg / Pos Ratio', 'Bitcoin Price (Adj Close)']]
df_Feb_reduced_Ccor.corr()


# In[29]:


# Regression analysis: Chen vs Bitcoin price
X = df_Feb_reduced['Chen Neg / Pos Ratio']
Y = df_Feb_reduced['Bitcoin Price (Adj Close)']
# with statsmodels
X = sm.add_constant(X) # adding a constant
 
model = sm.OLS(Y, X).fit()
predictions = model.predict(X) 
 
print_model = model.summary()
print(print_model)


# ## Pos / Neg Ratio & Bitcoin Price¶

# ### May 2021 Dataset

# #### VADER Analysis: 1-31 May

# In[30]:


# Plot VADER vs Bitcoin price
plt.scatter(df_May['VADER Pos / Neg Ratio'], df_May['Bitcoin Price (Adj Close)'], color='green')
plt.title('VADER Pos/Neg Ratio vs Bitcoin Price: 1-31 May', fontsize=14)
plt.xlabel('VADER Pos/Neg Ratio', fontsize=14)
plt.ylabel('Bitcoin Price', fontsize=14)
plt.grid(True)
plt.show()


# In[31]:


# Calculate correlation VADER vs Bitcoin price
df_May_Vcor = df_May[['VADER Pos / Neg Ratio', 'Bitcoin Price (Adj Close)']]
df_May_Vcor.corr()


# In[32]:


# Regression analysis: VADER vs Bitcoin price
X = df_May['VADER Pos / Neg Ratio']
Y = df_May['Bitcoin Price (Adj Close)']
# with statsmodels
X = sm.add_constant(X) # adding a constant
 
model = sm.OLS(Y, X).fit()
predictions = model.predict(X) 
 
print_model = model.summary()
print(print_model)


# #### VADER Analysis: 1-22 May

# In[33]:


# Plot VADER vs Bitcoin price
plt.scatter(df_May_reduced['VADER Pos / Neg Ratio'], df_May_reduced['Bitcoin Price (Adj Close)'], color='green')
plt.title('VADER Pos/Neg Ratio vs Bitcoin Price: 1-22 May', fontsize=14)
plt.xlabel('VADER Pos/Neg Ratio', fontsize=14)
plt.ylabel('Bitcoin Price', fontsize=14)
plt.grid(True)
plt.show()


# In[34]:


# Calculate correlation VADER vs Bitcoin price
df_May_reduced_Vcor = df_May_reduced[['VADER Pos / Neg Ratio', 'Bitcoin Price (Adj Close)']]
df_May_reduced_Vcor.corr()


# In[35]:


# Regression analysis: VADER vs Bitcoin price
X = df_May_reduced['VADER Pos / Neg Ratio']
Y = df_May_reduced['Bitcoin Price (Adj Close)']
# with statsmodels
X = sm.add_constant(X) # adding a constant
 
model = sm.OLS(Y, X).fit()
predictions = model.predict(X) 
 
print_model = model.summary()
print(print_model)


# #### Chen Analysis: 1-31 May

# In[36]:


# Plot Chen vs Bitcoin price
plt.scatter(df_May['Chen Pos / Neg Ratio'], df_May['Bitcoin Price (Adj Close)'], color='green')
plt.title('Chen Pos/Neg Ratio vs Bitcoin Price: 1-31 May', fontsize=14)
plt.xlabel('Chen Pos/Neg Ratio', fontsize=14)
plt.ylabel('Bitcoin Price', fontsize=14)
plt.grid(True)
plt.show()


# In[37]:


# Calculate correlation Chen vs Bitcoin price
df_May_Vcor = df_May[['Chen Pos / Neg Ratio', 'Bitcoin Price (Adj Close)']]
df_May_Vcor.corr()


# In[38]:


# Regression analysis: Chen vs Bitcoin price
X = df_May['Chen Pos / Neg Ratio']
Y = df_May['Bitcoin Price (Adj Close)']
# with statsmodels
X = sm.add_constant(X) # adding a constant
 
model = sm.OLS(Y, X).fit()
predictions = model.predict(X) 
 
print_model = model.summary()
print(print_model)


# #### Chen Analysis: 1-22 May

# In[39]:


# Plot Chen vs Bitcoin price
plt.scatter(df_May_reduced['Chen Pos / Neg Ratio'], df_May_reduced['Bitcoin Price (Adj Close)'], color='green')
plt.title('Chen Pos/Neg Ratio vs Bitcoin Price: 1-22 May', fontsize=14)
plt.xlabel('Chen Pos/Neg Ratio', fontsize=14)
plt.ylabel('Bitcoin Price', fontsize=14)
plt.grid(True)
plt.show()


# In[40]:


# Calculate correlation VADER vs Bitcoin price
df_May_reduced_Ccor = df_May_reduced[['Chen Pos / Neg Ratio', 'Bitcoin Price (Adj Close)']]
df_May_reduced_Ccor.corr()


# In[41]:


# Regression analysis: Chen vs Bitcoin price
X = df_May_reduced['Chen Pos / Neg Ratio']
Y = df_May_reduced['Bitcoin Price (Adj Close)']
# with statsmodels
X = sm.add_constant(X) # adding a constant
 
model = sm.OLS(Y, X).fit()
predictions = model.predict(X) 
 
print_model = model.summary()
print(print_model)


# ### Feb 2021 Dataset

# #### VADER Analysis: 1-28 Feb

# In[42]:


# Plot VADER vs Bitcoin price
plt.scatter(df_Feb['VADER Pos / Neg Ratio'], df_Feb['Bitcoin Price (Adj Close)'], color='green')
plt.title('VADER Pos/Neg Ratio vs Bitcoin Price: 1-28 Feb', fontsize=14)
plt.xlabel('VADER Pos/Neg Ratio', fontsize=14)
plt.ylabel('Bitcoin Price', fontsize=14)
plt.grid(True)
plt.show()


# In[43]:


# Calculate correlation VADER vs Bitcoin price
df_Feb_Vcor = df_Feb[['VADER Pos / Neg Ratio', 'Bitcoin Price (Adj Close)']]
df_Feb_Vcor.corr()


# In[44]:


# Regression analysis: VADER vs Bitcoin price
X = df_Feb['VADER Pos / Neg Ratio']
Y = df_Feb['Bitcoin Price (Adj Close)']
# with statsmodels
X = sm.add_constant(X) # adding a constant
 
model = sm.OLS(Y, X).fit()
predictions = model.predict(X) 
 
print_model = model.summary()
print(print_model)


# #### VADER Analysis: 1-20 Feb

# In[45]:


# Plot VADER vs Bitcoin price
plt.scatter(df_Feb_reduced['VADER Pos / Neg Ratio'], df_Feb_reduced['Bitcoin Price (Adj Close)'], color='green')
plt.title('VADER Pos/Neg Ratio vs Bitcoin Price: 1-20 Feb', fontsize=14)
plt.xlabel('VADER Pos/Neg Ratio', fontsize=14)
plt.ylabel('Bitcoin Price', fontsize=14)
plt.grid(True)
plt.show()


# In[46]:


# Calculate correlation VADER vs Bitcoin price
df_Feb_reduced_Vcor = df_Feb_reduced[['VADER Pos / Neg Ratio', 'Bitcoin Price (Adj Close)']]
df_Feb_reduced_Vcor.corr()


# In[47]:


# Regression analysis: VADER vs Bitcoin price
X = df_Feb_reduced['VADER Pos / Neg Ratio']
Y = df_Feb_reduced['Bitcoin Price (Adj Close)']
# with statsmodels
X = sm.add_constant(X) # adding a constant
 
model = sm.OLS(Y, X).fit()
predictions = model.predict(X) 
 
print_model = model.summary()
print(print_model)


# #### Chen Analysis: 1-28 Feb

# In[48]:


# Plot Chen vs Bitcoin price
plt.scatter(df_Feb['Chen Pos / Neg Ratio'], df_Feb['Bitcoin Price (Adj Close)'], color='green')
plt.title('Chen Pos/Neg Ratio vs Bitcoin Price: 1-28 Feb', fontsize=14)
plt.xlabel('Chen Pos/Neg Ratio', fontsize=14)
plt.ylabel('Bitcoin Price', fontsize=14)
plt.grid(True)
plt.show()


# In[49]:


# Calculate correlation Chen vs Bitcoin price
df_Feb_Ccor = df_Feb[['Chen Pos / Neg Ratio', 'Bitcoin Price (Adj Close)']]
df_Feb_Ccor.corr()


# In[50]:


# Regression analysis: Chen vs Bitcoin price
X = df_Feb['Chen Pos / Neg Ratio']
Y = df_Feb['Bitcoin Price (Adj Close)']
# with statsmodels
X = sm.add_constant(X) # adding a constant
 
model = sm.OLS(Y, X).fit()
predictions = model.predict(X) 
 
print_model = model.summary()
print(print_model)


# #### Chen Analysis: 1-20 Feb

# In[51]:


# Plot Chen vs Bitcoin price
plt.scatter(df_Feb_reduced['Chen Pos / Neg Ratio'], df_Feb_reduced['Bitcoin Price (Adj Close)'], color='green')
plt.title('Chen Pos/Neg Ratio vs Bitcoin Price: 1-20 Feb', fontsize=14)
plt.xlabel('Chen Pos/Neg Ratio', fontsize=14)
plt.ylabel('Bitcoin Price', fontsize=14)
plt.grid(True)
plt.show()


# In[52]:


# Calculate correlation Chen vs Bitcoin price
df_Feb_reduced_Ccor = df_Feb_reduced[['Chen Pos / Neg Ratio', 'Bitcoin Price (Adj Close)']]
df_Feb_reduced_Ccor.corr()


# In[53]:


# Regression analysis: Chen vs Bitcoin price
X = df_Feb_reduced['Chen Pos / Neg Ratio']
Y = df_Feb_reduced['Bitcoin Price (Adj Close)']
# with statsmodels
X = sm.add_constant(X) # adding a constant
 
model = sm.OLS(Y, X).fit()
predictions = model.predict(X) 
 
print_model = model.summary()
print(print_model)

