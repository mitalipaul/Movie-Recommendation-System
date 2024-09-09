#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


movies=pd.read_csv('tmdb_5000_movies.csv')
credits=pd.read_csv('tmdb_5000_credits.csv')


# In[3]:


movies.head()


# In[4]:


movies.keys()


# In[5]:


credits.head()


# In[6]:


credits.keys()


# In[7]:


credits.head(1)['cast'].values


# In[8]:


movies.shape


# In[9]:


credits.shape


# In[10]:


movies=movies.merge(credits,on='title')


# In[11]:


movies.shape


# In[12]:


movies['original_language'].value_counts()


# In[13]:


movies.info()


# In[14]:


#Required Attributes:
#genres
#id
#keywords
#title
#overview
#cast
#crew

movies=movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[15]:


movies.info()


# In[16]:


movies.head()


# In[17]:


movies.isnull().sum()


# In[18]:


movies.dropna(inplace=True)


# In[19]:


movies.isnull().sum()


# In[20]:


movies.duplicated().sum()


# In[21]:


movies.iloc[0]


# In[22]:


movies.iloc[0].genres


# In[23]:


import ast
def convert(obj):
    L=[]
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L


# In[24]:


def convertcast(obj):
    L=[]
    counter=0
    for i in ast.literal_eval(obj):
        if counter!=3:
            L.append(i['character'])
    return L


# In[25]:


def convertcrew(obj):
    L=[]
    for i in ast.literal_eval(obj):
        if i['job']=='Director':
            L.append(i['name'])
            break
    return L


# In[26]:


movies['genres']=movies['genres'].apply(convert)


# In[27]:


movies['keywords']=movies['keywords'].apply(convert)


# In[28]:


movies['cast']=movies['cast'].apply(convertcast)


# In[29]:


movies['crew']=movies['crew'].apply(convertcrew)


# In[30]:


movies['overview']=movies['overview'].apply(lambda x:x.split())


# In[31]:


movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])


# In[32]:


movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])


# In[33]:


movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])


# In[34]:


movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])


# In[35]:


movies.head()


# In[36]:


movies['tags']=movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']


# In[37]:


movies.head()


# In[38]:


df=movies[['movie_id','title','tags']]


# In[39]:


df.head()


# In[42]:


df['tags']=df['tags'].apply(lambda x:" ".join(x))


# In[43]:


df


# In[44]:


df['tags']=df['tags'].apply(lambda x:x.lower())


# In[45]:


df


# In[59]:


import nltk


# In[67]:


from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()


# In[68]:


def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)


# In[70]:


df['tags']=df['tags'].apply(stem)


# In[71]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000, stop_words='english')


# In[72]:


vectors=cv.fit_transform(df['tags']).toarray()


# In[73]:


vectors


# In[74]:


vectors.shape


# In[75]:


vectors[0]


# In[76]:


cv.get_feature_names_out()


# In[77]:


from sklearn.metrics.pairwise import cosine_similarity


# In[80]:


similarity=cosine_similarity(vectors)


# In[81]:


similarity.shape


# In[82]:


similarity[0]


# In[86]:


def recommend(movie):
    index = df[df['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])),reverse=True,key = lambda x: x[1])
    for i in distances[1:6]:
        print(df.iloc[i[0]].title)


# In[89]:


recommend('Batman Begins')


# In[ ]:




