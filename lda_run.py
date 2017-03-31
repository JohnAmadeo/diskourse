
# coding: utf-8

# In[1]:

import gensim
from gensim import corpora
import json
import time

start_time = time.time()

def print_time(label):
    print "-----------------------------"
    print label + ": " + str(time.time() - start_time) + " seconds"
    print "-----------------------------"

# In[2]:

with open('processed_corpus.json', 'r') as f:
    data = json.load(f)


# In[4]:

data = filter(None, data)
print "Corpus Size: " + str(len(data)) + " Articles"
# In[6]:

for i, article in enumerate (data):
    data[i] = [s.encode('utf-8') for s in article] # decode unicode str to str

# In[10]:

dictionary = corpora.Dictionary(data)
# Represents entire corpus in bag-of-words format i.e list of
 # (id, count) tuples
doc_term_matrix = [dictionary.doc2bow(doc) for doc in data]
print_time("Bag of Words")

# In[22]:

# Create variable to store LDA class
Lda = gensim.models.ldamodel.LdaModel 

# Train new model

# Initialize LDA object; estimates LDA model parameters on 
# corpus in bag-of-words format 
# lower probability threshold for more precise prediction
# ldamodel = Lda(doc_term_matrix, 
#                num_topics=15, 
#                id2word=dictionary, 
#                passes=200, 
#                minimum_probability = 0.0001) 
# print_time("Train LDA model")
# ldamodel.save('10ap.model')

# Load old model
ldamodel = Lda.load('10ap.model')

# In[25]:

# result = []
# for i in range(len(data[:10])):
#     article = dict((x,y) for x,y in ldamodel[dictionary.doc2bow(data[:10][i])])
#     for h in range (0, 10):
#         if h not in article.keys():
#             article[h] = 0
#     result.append(article)


# In[26]:

# Print word distribution (top words only) and corresponding percentage of each topic
print("----------- Word distribution for each topic -----------")
topics = ldamodel.print_topics(num_topics=15, num_words=5)
topics = [dict((item.split("*\"")[1][:-1].encode("utf8"),item.split("*\"")[0].encode("utf8"))  for item in topic[1].split(" + ")) for topic in topics]

for idx, topic in enumerate(topics):
    word_dist = topic
    print "Topic " + str(idx) 
    word_dist_string = [key + ": " + str(word_dist[key]) 
                        for key in word_dist.keys()]
    print " ".join(word_dist_string)
    print

# Print topic distribution for first 5 articles
print("----------- Topic distribution for first 10 articles -----------")
clean_corpus = []
with open('clean_corpus.json', 'r') as f:
    clean_corpus = json.load(f)

for idx, article in enumerate(clean_corpus[:10]):
    print "--------- Article " + str(idx) + "-------------"
    print article[:300]
    print '--------- Topic Distribution --------------'
    for idx, topic in enumerate(ldamodel[doc_term_matrix[idx]]):
        topic_id = topic[0]
        topic_dist = topic[1]
        topic_word_dist = topics[idx]
        topic_word_dist_string = [key + ": " + str(topic_word_dist[key]) 
                                  for key in topic_word_dist.keys()]

        print "Topic " + str(topic[0]) + ": " + str(round(topic[1], 5))
        print " ".join(topic_word_dist_string)
        print 

    print '\n\n\n'

# Write document similarity function 
# (see http://stackoverflow.com/questions/22433884/python-gensim-how-to-calculate-document-similarity-using-the-lda-model)

print_time("Total")
# In[ ]:



