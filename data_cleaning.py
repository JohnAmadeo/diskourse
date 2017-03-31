
# coding: utf-8

# In[10]:

import json
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from collections import Counter
import pandas as pd
import gzip
import string
import time

start_time = time.time()
# In[13]:

# read gz file and get corpus: list of list of articles
with gzip.open('ap.gz', 'rb') as f:
    file_content = f.read()

file_content = file_content.decode("utf-8")

data = file_content.split("</DOCNO>")
corpus = []
for i, item in enumerate (data):
    if i in range (1,2249):
        article = item.split("\n<TEXT>\n")[1].split("\n </TEXT>\n")[0]
        corpus.append(article)

# In[67]:

# filter empty article
corpus = filter(None, corpus)
# corpus = ["The wounded teacher, 37-year-old Sam Marino, was in serious condition Saturday with gunshot wounds in the shoulder. Police said the boy also shot at a third teacher, Susan Allen, 31, as she fled from the room where Marino was shot. He then shot Marino again before running to a third classroom where a Bible class was meeting. The youngster shot the glass out of a locked door before opening fire, police spokesman Lewis Thurston said. When the youth's pistol jammed, he was tackled by teacher Maurice Matteson, 24, and other students, Thurston said. ``Once you see what went on in there, it's a miracle that we didn't have more people killed,'' Police Chief Charles R. Wall said. Police didn't have a motive, Detective Tom Zucaro said, but believe the boy's primary target was not a teacher but a classmate."]
corpus = corpus[:5]

# In[68]:

# initialized lemmatizer tools, get a list of bad tags and a list of stopwords
lemma = WordNetLemmatizer()
bad_tags = ['CD', 'MD', 'IN', 'DT', 'ABL', 'ABN', 'AP', 'CC', 'CS', 'PN', 'POS', 'PP$', 'PPSS', 'PPSS+BER', 'PPSS+MD', 'QL', 'UH', 'WDT', 'WPS', 'WRB']
#n't is a stop word because
stop = ["a","able","about","above","abst","accordance","according","accordingly","across","act","actually","added","adj","affected","affecting","affects","after","afterwards","again","against","ah","all","almost","alone","along","already","also","although","always","am","among","amongst","an","and","announce","another","any","anybody","anyhow","anymore","anyone","anything","anyway","anyways","anywhere","apparently","approximately","are","aren","arent","arise","around","as","aside","ask","asking","at","auth","available","away","awfully","b","back","be","became","because","become","becomes","becoming","been","before","beforehand","begin","beginning","beginnings","begins","behind","being","believe","below","beside","besides","between","beyond","biol","both","brief","briefly","but","by","c","ca","came","can","cannot","can't","cause","causes","certain","certainly","co","com","come","comes","contain","containing","contains","could","couldnt","d","date","did","didn't","different","do","does","doesn't","doing","done","don't","down","downwards","due","during","e","each","ed","edu","effect","eg","eight","eighty","either","else","elsewhere","end","ending","enough","especially","et","et-al","etc","even","ever","every","everybody","everyone","everything","everywhere","ex","except","f","far","few","ff","fifth","first","five","fix","followed","following","follows","for","former","formerly","forth","found","four","from","further","furthermore","g","gave","get","gets","getting","give","given","gives","giving","go","goes","gone","got","gotten","h","had","happens","hardly","has","hasn't","have","haven't","having","he","hed","hence","her","here","hereafter","hereby","herein","heres","hereupon","hers","herself","hes","hi","hid","him","himself","his","hither","home","how","howbeit","however","hundred","i","id","ie","if","i'll","im","immediate","immediately","importance","important","in","inc","indeed","index","information","instead","into","invention","inward","is","isn't","it","itd","it'll","its","itself","i've","j","just","k","keep  keeps","kept","kg","km","know","known","knows","l","largely","last","lately","later","latter","latterly","least","less","lest","let","lets","like","liked","likely","line","little","'ll","look","looking","looks","ltd","m","made","mainly","make","makes","many","may","maybe","me","mean","means","meantime","meanwhile","merely","mg","might","million","miss","ml","more","moreover","most","mostly","mr","mrs","much","mug","must","my","myself","n","na","name","namely","nay","nd","near","nearly","necessarily","necessary","need","needs","neither","never","nevertheless","new","next","nine","ninety","no","nobody","non","none","nonetheless","noone","nor","normally","nos","not","noted","nothing","now","nowhere","n't", "o","obtain","obtained","obviously","of","off","often","oh","ok","okay","old","omitted","on","once","one","ones","only","onto","or","ord","other","others","otherwise","ought","our","ours","ourselves","out","outside","over","overall","owing","own","p","page","pages","part","particular","particularly","past","per","perhaps","placed","please","plus","poorly","possible","possibly","potentially","pp","predominantly","present","previously","primarily","probably","promptly","proud","provides","put","q","que","quickly","quite","qv","r","ran","rather","rd","re","readily","really","recent","recently","ref","refs","regarding","regardless","regards","related","relatively","research","respectively","resulted","resulting","results","right","run","s","said","same","saw","say","saying","says","sec","section","see","seeing","seem","seemed","seeming","seems","seen","self","selves","sent","seven","several","shall","she","shed","she'll","shes","should","shouldn't","show","showed","shown","showns","shows","significant","significantly","similar","similarly","since","six","slightly","so","some","somebody","somehow","someone","somethan","something","sometime","sometimes","somewhat","somewhere","soon","sorry","specifically","specified","specify","specifying","still","stop","strongly","sub","substantially","successfully","such","sufficiently","suggest","sup","sure  t","take","taken","taking","tell","tends","th","than","thank","thanks","thanx","that","that'll","thats","that've","the","their","theirs","them","themselves","then","thence","there","thereafter","thereby","thered","therefore","therein","there'll","thereof","therere","theres","thereto","thereupon","there've","these","they","theyd","they'll","theyre","they've","think","this","those","thou","though","thoughh","thousand","throug","through","throughout","thru","thus","til","tip","to","together","too","took","toward","towards","tried","tries","truly","try","trying","ts","twice","two","u","un","under","unfortunately","unless","unlike","unlikely","until","unto","up","upon","ups","us","use","used","useful","usefully","usefulness","uses","using","usually","v","value","various","'ve","very","via","viz","vol","vols","vs","w","want","wants","was","wasnt","way","we","wed","welcome","we'll","went","were","werent","we've","what","whatever","what'll","whats","when","whence","whenever","where","whereafter","whereas","whereby","wherein","wheres","whereupon","wherever","whether","which","while","whim","whither","who","whod","whoever","whole","who'll","whom","whomever","whos","whose","why","widely","willing","wish","with","within","without","wont","words","world","would","wouldnt","www","x","y","yes","yet","you","youd","you'll","your","youre","yours","yourself","yourselves","you've","z","zero"]
punctuation = set(string.punctuation)
punctuation.update(["''", '``'])

# In[69]:


for i in range (len(corpus)):
    # tokenize the word
    corpus[i] = word_tokenize(corpus[i])
    # remove stop word
    corpus[i] = [word for word in corpus[i] if word.lower() not in stop]
    # lemmatize all words
    corpus[i] = [lemma.lemmatize(word).encode("utf8") for word in corpus[i]]
    # remove words associated with bad_tags
    corpus[i] = [tuple[0].lower() for tuple in pos_tag(corpus[i]) if tuple[1] not in bad_tags]
    # remove punctuation
    corpus[i] = [word for word in corpus[i] if word not in punctuation]
    # corpus[i] = " ".join(corpus[i]).translate (None, string.punctuation)
    # corpus[i] = word_tokenize(corpus[i])

    print "Document " + str(i) + " " + str(time.time() - start_time) + " seconds"
    print corpus[i]
    print "-----------------------------"


exit(0)
# In[72]:

# count # of word occurenences
corpus_flat = [item for f in range(len(corpus)) for item in corpus[f]]
d = dict(Counter(corpus_flat))
df= pd.DataFrame(d.items()) 
df.columns = ["word", "count"]
df.sort_values(by = "count", ascending = False)


# In[76]:

# write cleaned data into a json
with open('cleaned_article.json', 'wb') as outfile:
    json.dump(corpus, outfile)
