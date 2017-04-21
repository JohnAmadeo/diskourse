import gensim
from gensim import corpora
import json
from customprint import print_time, print_blank, print_line

with open('processed_corpus.json', 'r') as f:
    corpus = json.load(f)


corpus = filter(None, corpus)
print "Corpus Size: " + str(len(corpus)) + " Articles"

for i, article in enumerate (corpus):
    corpus[i] = [s.encode('utf-8') for s in article] # decode unicode str to str


dictionary = corpora.Dictionary(corpus)
# Represents entire corpus in bag-of-words format i.e list of
 # (id, count) tuples
doc_term_matrix = [dictionary.doc2bow(doc) for doc in corpus]
print_time("Created bag-of-words representation of corpus")


# Create variable to store LDA class
Lda = gensim.models.ldamodel.LdaModel 
MatrixSimilarity = gensim.similarities.MatrixSimilarity

# Train new model

# Initialize LDA object; estimates LDA model parameters on 
# corpus in bag-of-words format 
# lower probability threshold for more precise prediction
# ldamodel = Lda(doc_term_matrix, 
#                num_topics=15, 
#                id2word=dictionary, 
#                passes=200, # ~17 minutes
#                minimum_probability = 0.0001) 
# print_time("Trained LDA model")
# ldamodel.save('ap_lda.model')

# Load old model
ldamodel = Lda.load('ap_lda.model')

# Train similarity index (~8 seconds w/ ~2200 documents)
# sim_index = MatrixSimilarity(ldamodel[doc_term_matrix])
# sim_index.save("ap_similarity.index")
# print_time("Created similarity index based on corpus")

# Load old similarity index
sim_index = MatrixSimilarity.load('ap_similarity.index')









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
    print_blank


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
print_line('Show similarity for first 10 articles')

for i in range(10):
    article_lda = ldamodel[doc_term_matrix[i]]
    # List of (article index in corpus, similarity score) pairs
    similar_documents = sorted(enumerate(sim_index[article_lda]),
                               key=lambda pair: pair[1],
                               reverse=True)

    print clean_corpus[i]
    print_blank(3)

    print_line('Most Similar Articles')
    for idx, score in similar_documents[:5]:
        print score
        print clean_corpus[idx][:300]
        print_blank(2)

    print_line('Least Similar Articles')
    for idx, score in similar_documents[-5:]:
        print score
        print clean_corpus[idx][:300]
        print_blank(2)


print_time("File lda_run.py executed")


