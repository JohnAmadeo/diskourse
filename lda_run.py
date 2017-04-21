import gensim
from gensim import corpora
import json
# import time
from customprint import print_time, print_blank

# start_time = time.time()

# def print_time(label):
#     print "----------------------------------------------------------"
#     print label + ": " + str(time.time() - start_time) + " seconds"
#     print "----------------------------------------------------------"

with open('processed_corpus.json', 'r') as f:
    data = json.load(f)


data = filter(None, data)
print "Corpus Size: " + str(len(data)) + " Articles"

for i, article in enumerate (data):
    data[i] = [s.encode('utf-8') for s in article] # decode unicode str to str


dictionary = corpora.Dictionary(data)
# Represents entire corpus in bag-of-words format i.e list of
 # (id, count) tuples
doc_term_matrix = [dictionary.doc2bow(doc) for doc in data]
print_time("Bag-of-words representation created")


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

# result = []
# for i in range(len(data[:10])):
#     article = dict((x,y) for x,y in ldamodel[dictionary.doc2bow(data[:10][i])])
#     for h in range (0, 10):
#         if h not in article.keys():
#             article[h] = 0
#     result.append(article)


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

for i, article_text in enumerate(clean_corpus[:10]):
    print "--------- Article " + str(i) + "-------------"
    print article_text[:150]

    print '--------- Topic Distribution --------------'
    # sort topics by centrality to article's theme
    article_lda = sorted(ldamodel[doc_term_matrix[i]], 
                         key=lambda topic: topic[1],
                         reverse=True)

    for j, topic in enumerate(article_lda):
        topic_id = topic[0]
        topic_dist = topic[1]
        topic_word_dist = topics[j]
        topic_word_dist_string = [key + ": " + str(topic_word_dist[key]) 
                                  for key in topic_word_dist.keys()]

        print "Topic " + str(topic[0]) + ": " + str(round(topic_dist, 5))
        print " ".join(topic_word_dist_string)
        print_blank(1) 

    print_blank(3)

# Write document similarity function 
# (see http://stackoverflow.com/questions/22433884/python-gensim-how-to-calculate-document-similarity-using-the-lda-model)

print_time("File lda_run.py executed")



