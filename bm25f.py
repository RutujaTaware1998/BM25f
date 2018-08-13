

import pandas as pd
import re
import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import numpy as np


df = pd.read_csv('title_body1.csv')
title_weight = 6
body_weight = 1

#convert to lower case
df['title_lower'] = df['title'].apply(lambda x : " ".join( x.lower() for x in x.split()))
df['body_lower'] = df['body'].apply(lambda x : " ".join( x.lower() for x in x.split()))



#remove punctuations
df['title_lower'] = df['title_lower'].str.replace('[^\w\s\d]' ,' ')
df['body_lower'] = df['body_lower'].str.replace('[^\w\s\d]' ,' ')


#tokenizing

df['body_token'] = df['body_lower'].apply(lambda x : word_tokenize(x))
df['title_token'] = df['title_lower'].apply(lambda x : word_tokenize(x))


#removing stopwords
stop = stopwords.words('english')
df['title_token'] = df['title_token'].apply(lambda x : list(i for i in x if i not in stop))
df['body_token'] = df['body_token'].apply(lambda x : list(i for i in x if i not in stop))


#stemming
stemmer = PorterStemmer()
df['title_token'] =  df['title_token'].apply(lambda x : list(stemmer.stem(i)  for i in x))
df['body_token'] = df['body_token'].apply(lambda x : list(stemmer.stem(i)  for i in x))    


#Creating new unstructured document
df['unstructured'] = df['title_token']*title_weight + df['body_token']*body_weight
#calculating avg doc length
total_length = df['unstructured'].apply(len).sum(axis = 0)
total_articles = df.shape[0]
avg_len = total_length*1.0 / total_articles

                 
                 
print('enter the query')
query = raw_input()    
#clean the query
query = query.lower()
query = re.sub('[^\w\s\d]' ,' ' , query)
words = word_tokenize(query)
words = [i for i in words if i not in stop] 
words = [stemmer.stem(i) for i in words]


df['relevance'] = 0

for word in words:
        r = df['unstructured'].apply(lambda x:word in x).sum(axis = 0)
        

        idf = np.log((total_articles*1.0)/(r + 1) ) + 1  
        

        def calc(doc):
            k = 1.2
            b =0.75
            tf = doc.count(word)

            weight = idf * ((k+1)*tf )/(k*(1 - b + b*(len(doc) / avg_len)) + tf)
            return weight  
            
            
        df['relevance'] = df['relevance'] + df['unstructured'].apply(lambda x : calc(x))

final_df = df.loc[: ,['title' , 'body' , 'relevance']]
final_df.to_csv('search_results1.csv')

"""

For testing:

Test Case 1)enter query:
            eyes
Test Case 2)enter query:
            eyes and  vision
Test Case 3)enter query:
            ears            

"""




