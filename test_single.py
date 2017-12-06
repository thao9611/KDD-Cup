import pickle
import data_io
import pandas as pd
import numpy as np
from feature_set import *
from Thao_features import *
from kamil_features import *
import time
import unidecode

def read_all_DS():
    dataset = {}

    dataset['paper'] = pd.read_csv('dataRev2/Paper.csv')
    dataset['author'] = pd.read_csv('dataRev2/Author.csv')
    '''
    dataset['conference'] = pd.read_csv('dataRev2/Conference.csv')
    dataset['journal'] = pd.read_csv('dataRev2/Journal.csv')
    '''
    dataset['paper_author'] = pd.read_csv('dataRev2/PaperAuthor.csv')
    return dataset

start_time = time.time()
print ("Read dataset")
dataset = read_all_DS()

paper_author = dataset['paper_author']
paper_author["Name"].fillna("", inplace=True)
paper_author["Affiliation"].fillna("", inplace=True)

author = dataset['author']
author["Name"].fillna("", inplace=True)
author["Affiliation"].fillna("", inplace=True)

paper = dataset['paper']
paper["Title"].fillna("", inplace=True)

print ("Indexing start %s" % (time.time() - start_time))
pa_indexed = paper_author.set_index(['AuthorId','PaperId']).sort_index()
author_indexed = author.set_index(['Id']).sort_index()
paper_indexed = paper.set_index(['Id']).sort_index()

'''
print ("Start loop")
for index, row in paper_author.iterrows():
    aid = row['AuthorId']
    name = row['Name'].lower()


    target = ["mayeng", "sung", "hyon"]
    flag = False
    for t in target:
        if t in name:
            flag = True
            break
    if (flag):
        print (aid, name)
'''

validset = pd.read_csv('dataRev2/Valid.csv')
for index, row in validset.iterrows():
    aid = row['AuthorId']
    name = author_indexed.loc[aid]["Name"].lower()
    if ("kim" in name):
        print (aid, name)



print ("Indexing ends %s" % (time.time() - start_time))
print("Give author id")
while(True):
    aid = int(input())
    if (aid == 0):
        break;
    while (aid not in author_indexed.index):
        print("Author id not exists, give author id agian")
        aid = int(input())
    print(author_indexed.loc[aid]["Name"])
    print(author_indexed.loc[aid]["Affiliation"])


#print(author_indexed.loc[aid]["Name"])
#print(author_indexed.loc[aid]["Affiliation"])

print("Give paper id")
while(True):
    pid = int(input())
    if (pid == 0):
        break
    while (pid not in paper_indexed.index):
        print("Paper id not exists, give paper id agian")
        pid = int(input())
    print(paper_indexed.loc[pid]["Title"])