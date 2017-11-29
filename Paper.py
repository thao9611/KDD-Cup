
# coding: utf-8

# In[1]:

import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
paper = pd.read_csv(r"C:\Users\Admin\Downloads\kdd cup\dataRev\Paper.csv")
paper= paper.set_index("Id")
paper['Keyword']= paper['Keyword'].fillna("")
paper['Title']= paper['Title'].fillna("")
title = list(paper["Title"])
paper.head()


# In[9]:


stopword = set(stopwords.words('english'))
porter = PorterStemmer()
def tokenize(text):
    #text = text.split() # get single words 
    #table = maketrans('','',string.punctuation)
    #stripped = [w.translate(table).lower() for w in text]#get rid of all punctuation
    words = word_tokenize(text) #split words
    words = [w.lower() for w in words if w.isalpha()] #get rid of punctuation
    words =[w for w in words if  not w in stopword]
    stemmed = [porter.stem(w) for w in words]
    
    
    return stemmed
paper['Token'] = paper.Title.map(tokenize)
paper['Token'][:5]
    
       


# In[ ]:


count = CountVectorizer()
tfidf = TfidfTransformer()
count_token =count.fit_transform(title)
#tfid_token = tfidf.fit_transform(count_token)
threshold = 20




# In[ ]:

count.vocablary_


# In[ ]:




# In[ ]:



