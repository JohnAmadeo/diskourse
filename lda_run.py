
# coding: utf-8

# In[1]:

import gensim
from gensim import corpora
import json


# In[2]:

with open('cleaned_article.json', 'r') as f:
     data = json.load(f)


# In[4]:

data = filter(None, data)


# In[6]:

for i, article in enumerate (data):
    data[i] = [s.encode('utf-8') for s in article] # decode unicode str to str


# In[10]:

dictionary = corpora.Dictionary(data)
# Represents entire corpus in bag-of-words format i.e list of
 # (id, count) tuples
doc_term_matrix = [dictionary.doc2bow(doc) for doc in data]


# In[22]:

# Create variable to store LDA class
Lda = gensim.models.ldamodel.LdaModel 
# Initialize LDA object; estimates LDA model parameters on 
# corpus in bag-of-words format 
ldamodel = Lda(doc_term_matrix, num_topics=10, id2word=dictionary, passes=50, minimum_probability = 0.0001) # lower probability threshold for more precise prediction
# Save model
ldamodel.save('10ap.model')


# In[23]:

# Print word distribution (top words only) of each topic
print("----------- Word distribution for each topic -----------")
topics = ldamodel.print_topics(num_topics=10, num_words=20)
for topic in topics:
    dist = [item.split("*\"")[1][:-1].encode("utf8") for item in topic[1].split(" + ")]
    print(dist)


# In[25]:

result = []
for i in range(len(data[:10])):
    article = dict((x,y) for x,y in ldamodel[dictionary.doc2bow(data[:10][i])])
    for h in range (0, 10):
        if h not in article.keys():
            article[h] = 0
    result.append(article)


# In[26]:

result


# In[38]:

# Print word distribution (top words only) and correspondence percentage of each topic
print("----------- Word distribution for each topic -----------")
topics = ldamodel.print_topics(num_topics=10, num_words=5)
for topic in topics:
    dist = dict((item.split("*\"")[1][:-1].encode("utf8"),item.split("*\"")[0].encode("utf8"))  for item in topic[1].split(" + "))
    print(dist)


# In[ ]:



