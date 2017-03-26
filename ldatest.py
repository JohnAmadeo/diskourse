from nltk.corpus import stopwords
from nltk.corpus import brown
from nltk.stem.wordnet import WordNetLemmatizer
import string
import gensim
from gensim import corpora
import time
import random

def main():
    start_time = time.time()

    # doc1 = "Sugar is bad to consume. My sister likes to have sugar, but not my father."
    # doc2 = "My father spends a lot of time driving my sister around to dance practice."
    # doc3 = "Doctors suggest that driving may cause increased stress and blood pressure."
    # doc4 = "Sometimes I feel pressure to perform well at school, but my father never seems to drive my sister to do better."
    # doc5 = "Health experts say that Sugar is not good for your lifestyle."

    # doc_complete = [doc1, doc2, doc3, doc4, doc5] 

    # Array of articles from brown corpus (each article is list of tokens)
    corpus = [brown.words(doc_id) for doc_id in brown.fileids()]
    # Shuffle order of array so articles from same categories are not
    # sequentially placed next to each other
    random.shuffle(corpus)

    doc_complete = corpus[:100]
    # Cleans up each document and then tokenizes them
    doc_clean = [clean(doc).split() for doc in doc_complete]
        # Creates a dictionary object with the document corpus
    dictionary = corpora.Dictionary(doc_clean)
    # Represents entire corpus in bag-of-words format i.e list of
    # (id, count) tuples
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
    # Create variable to store LDA class
    Lda = gensim.models.ldamodel.LdaModel
    # Initialize LDA object; estimates LDA model parameters on 
    # corpus in bag-of-words format 
    ldamodel = Lda(doc_term_matrix, num_topics=5, id2word=dictionary, passes=30)
    # Save model
    ldamodel.save('100brown.model')

    # Print word distribution (top words only) of each topic
    print("----------- Word distribution for each topic -----------")
    topics = ldamodel.print_topics(num_topics=5, num_words=10)
    for topic in topics:
        dist = [item.split("*\"")[1][:-1] for item in topic[1].split(" + ")]
        print(dist)

    # Print topic distribution of select documents
    print("----------- Topic distribution for select documents -----------")

    # for distribution in [ldamodel[" ".join(doc)]for doc in corpus[:5]]:
    #     print (distribution)

    print_runtime(start_time)

def print_runtime(start_time):
    print("\n------ %s seconds ------" % (time.time() - start_time))

def clean(doc):
    """
    Takes in a list of tokens, removes stop words and punctuation and
    lemmatizes all remaining words in the string, and returns it
    """
    stop = set(stopwords.words('english'))
    exclude = set(string.punctuation)
    lemma = WordNetLemmatizer()

    lower = [token.lower() for token in doc]

    stop_free = " ".join([token for token in lower if token not in stop])
    punc_free = ''.join(char for char in stop_free if char not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

if __name__ == "__main__":
    main()

