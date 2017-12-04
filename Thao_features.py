from collections import defaultdict
import pandas as pd
import numpy
import string
import pickle
import data_io
import time
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import 
stopword = set(stopwords.words('english'))
porter = PorterStemmer()

#frequency of a author-paper pair in PaperAuthor.csv
def author_paper_frequency_count(dataset, author_paper_pairs):
    author_paper_count = defaultdict(int)
    ap_1 = dataset['ap_duplicate']

    for i in author_paper_pairs:
        author_paper_count[i] = ap_1.loc[i, "Affiliation"]
    return author_paper_count

def process_aff(text):
    for i in string.punctuation: text = text.replace(i,' ')
    words = word_tokenize(text) #split words
    words = [w.lower() for w in words if w.isalpha()] #get rid of punctuation
    words = [w for w in words if w not in list(stopword)]
    words = [w for w in words if w not in  ["institute","univ",'university',"college","department","science","technology",
                                            "de","engineer","lab","dept","falcuty",] ]
    stemmed = [porter.stem(w) for w in words]
    return stemmed
def author_affiliation(data):
    pa = data["paper_author"]
    affiliation = defaultdict(str)
    
    pa['Affiliation'] = pa['Affiliation'].fillna("")
    pa_1 = pd.DataFrame(pd.pivot_table(pa, values = "Affiliation",index = ["AuthorId"], aggfunc = "sum"))
    author = list(pa_1.index)
    for i in author:
        affiliation[i]= process_aff(pa_1.loc[i,"Affiliation"])
    return affiliation

# more efficient approach 
def target_author_and_coauthor_of_target_paper_by_affiliation(dataset,author_paper_pairs):
    pa = dataset["paper_author"]
    aff = author_affiliation(dataset)
    author_sim = defaultdict(int)
    pa= pa.set_index("PaperId")
    for i in author_paper_pairs:
        coauthor= list(pa.loc[i[1], "AuthorId"])
        author_sim[i] = sum(common_word(aff[i[0]],aff[j]) for j in coauthor)
    return author_sim

#for keyword of Papers
def filter_keyword(text):
    for i in string.punctuation: text = text.replace(i,' ')
    words = word_tokenize(text) #split words
    words = [w.lower() for w in words if w.isalpha()] #get rid of punctuation
    words = [w for w in words if w not in  ["keywords"] ]
    stemmed = [porter.stem(w) for w in words]
    return stemmed

# for title of papers
def tokenize(text):
    words = word_tokenize(text) #split words
    words = [w.lower() for w in words if w.isalpha()] #get rid of punctuation
    words =[w for w in words if  not w in stopword]
    stemmed = [porter.stem(w) for w in words]
    return stemmed

#return the keywords of each paper
def paper_keywords(data):
    paper = data['paper']
    paperid = list(paper["Id"])
    paper_keyword = defaultdict(list)

    paper= paper.set_index("Id")
    paper['Keyword']= paper['Keyword'].fillna("")
    paper['Title']= paper['Title'].fillna("")

    title = list(paper["Title"])
    cnt = 0
    start_time = time.time()
    titleTokens = []

    print("Start title!!!")
    for t in title:
        cnt += 1
        if (cnt % 100000 == 0):
            print("Count: ", cnt)
            print("Time: ", time.time() - start_time)
        titleTokens.append(tokenize(t))
    paper['Token'] = titleTokens


    #paper['Token'] = paper.Title.map(tokenize)
    print ("Start keyword!!!")

    keywords = list(paper['Keyword'])
    cnt2 = 0
    keywordTokens = []
    for k in keywords:
        cnt2 += 1
        if (cnt2 % 100000 == 0):
            print("Count: ", cnt)
            print("Time: ", time.time() - start_time)
        keywordTokens.append(filter_keyword(k))

    paper['Keyword_pro'] = keywordTokens

    print("Start concatenation!!!")

    #TODO: change all "apply", "map" functions to explicit for loop. Don't use "for loop" in the list because it causes memory limit error.

    #concatenate keyword and token
    paper['Key_token'] = paper[['Keyword_pro','Token']].apply((lambda x: ' '.join(list(set([i for z in x for i in z])))), axis =1)
    token = list(paper['Key_token'])
    count = CountVectorizer(min_df = 5) #only take words with df > 5
    tfidf = TfidfTransformer()
    count_token =count.fit_transform(token).toarray() #2000*527
    
    vocab = list(count.vocabulary_.keys())
    #list of common words in each title of each document
    paper['Common word'] = paper['Key_token'].map(lambda x: [i for i in x.split() if i in vocab])
    for i in paperid:
        paper_keyword[i] = paper.loc[i,'Common word']

    pickle.dump(paper_keyword, open(data_io.get_paths()["paper_title_tokens"], 'wb'))
    return paper_keyword

#how similar two documents are based on keywords

def paper_common_word(tokens, id1, id2):
    paper_keyword = tokens
    sim = 0
    word1 = paper_keyword[id1]
    word2 = paper_keyword[id2]
    

def common_word(word1, word2):
    for i in word1:
        if i in word2:
            sim += 1
    return sim
    
def target_paper_and_papers_of_target_author_by_keywords(dataset, author_paper_pairs):
    paper_sim = defaultdict(int)
    trainset = dataset['paper_author']


    trainset = trainset.set_index('AuthorId')

    tokens = pickle.load(open(data_io.get_paths()["paper_title_tokens"], 'rb'))
   
    for ap in author_paper_pairs:
        target_author_papers= list(trainset.loc[ap[0], "PaperId"])
        paper_sim[ap] = sum(paper_common_word(tokens, ap[1], pid) for pid in target_author_papers)
    return paper_sim

def paper_year(data,id1, id2):
    paper = data['paper']
    paper= paper.set_index("Id")
    paper['Year'] = paper['Year'].fillna(0)
    year1 = paper.loc[id1,'Year']
    year2 = paper.loc[id2, 'Year']
    if (year1 == 0 | year2 == 0 | year1 == -1 | year2 == -1 ): return 0
    return 1+ abs(year1 - year2 )

def target_paper_and_papers_of_target_author_by_years(dataset, author_paper_pairs):
    paper_sim = defaultdict(int)
    trainset = dataset['paper_author']
    trainset  = trainset.set_index("AuthorId")
    for i in author_paper_pairs:
        trained_paper= trainset.loc[i[0], "PaperId"]
        total_year = 0
        num_of_paper = 0
        for j in trained_paper:
            a = paper_year(dataset, i[1],j)
            if (a > 0):
                total_year += a
                num_of_paper += 1
        if(num_of_paper == 0): paper_sim[i] = 0
        paper_sim[i] = total_year/num_of_paper
    return paper_sim
