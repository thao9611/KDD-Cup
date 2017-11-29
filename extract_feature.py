import pickle
import data_io
import pandas as pd
import numpy as np
from feature_set import *
from Thao_features import *

def read_all_DS():
    dataset = {}
    dataset['paper'] = pd.read_csv('C:/Users/Admin/Downloads/KDD Cup/dataRev2/Paper.csv')
    dataset['author'] = pd.read_csv('C:/Users/Admin/Downloads/KDD Cup/dataRev2/Author.csv')
    dataset['conference'] = pd.read_csv('C:/Users/Admin/Downloads/KDD Cup/dataRev2/Conference.csv')
    dataset['journal'] = pd.read_csv('C:/Users/Admin/Downloads/KDD Cup/dataRev2/Journal.csv')
    dataset['paper_author'] = pd.read_csv('C:/Users/Admin/Downloads/KDD Cup/dataRev2/PaperAuthor.csv')
    return dataset
# return a list of paper id
def parse_paper_ids(paper_ids_string):
    return paper_ids_string.strip().split()
# return  list of pair of author and his/her paper lists
def parse_targetset(targetset):
    pair_list = []
    author_id_list = targetset['AuthorId']

    for i in range(len(author_id_list)):
        author_id = author_id_list[i]
        papers = targetset[targetset.AuthorId == author_id]['PaperIds'].unique()[0]
        papers = parse_paper_ids(papers)
        for j in range(len(papers)):
            paper_id = int(papers[j])
            pair_list.append( (author_id, paper_id) )
    return list(set(pair_list))

def generate_feature_list(author_paper_pairs, ap_to_feature_list):
    result_list = []

    temp_dict = {} # { (author, paper) => [f1, f2 ...] }
    for ap_pair in author_paper_pairs:
        temp_dict[ap_pair] = []

    for i in range(len(ap_to_feature_list)):
        feature_dict = ap_to_feature_list[i]
        for ap_pair in author_paper_pairs:
            feature = feature_dict[ap_pair]
            temp_dict[ap_pair].append(feature)

    for key in temp_dict.keys():
        result_list.append(key + tuple( temp_dict[key] ))

    return result_list

def get_features(dataset, targetset):
    author_paper_pairs = parse_targetset(targetset)

    # Keep the format of f# (dictionary): { (a1, p1): feature_value1, (a2, p2): feature_value2 ... }
    # Add your features here and add them to feature_list!
    harry_f1 = get_author_publishes_how_many_paper_in_PaperAuthor(dataset, author_paper_pairs)
    harry_f2 = get_paper_has_how_many_author_in_PaperAuthor(dataset, author_paper_pairs)
    thao_f1 = author_paper_frequency_count(dataset)
    thao_f2 = author_paper_affiliation(dataset)
    thao_f3 = target_paper_and_confirmed_papers_of_target_author_by_keywords(dataset,author_paper_pairs)
    thao_f4 = target_paper_and_deleted_papers_of_target_author_by_keywords(dataset,author_paper_pairs)
    thao_f5 = target_paper_and_confirmed_papers_of_target_author_by_years(dataset, author_paper_pairs)


    harry_list = [harry_f1, harry_f2]
    kamil_list = []
    thao_list = [thao_f1, thao_f2,thao_f3,thao_f4, thao_f5]
    feature_list = harry_list + kamil_list + thao_list

    result_list = generate_feature_list(author_paper_pairs, feature_list)
    return result_list

def main():
    print("Reading csv files")
    dataset = read_all_DS()

    trainset = pd.read_csv('dataRev2/Train.csv')
    train_confirmed = trainset[['AuthorId', 'ConfirmedPaperIds']].rename(columns = {'ConfirmedPaperIds':'PaperIds'})
    train_deleted = trainset[['AuthorId', 'DeletedPaperIds']].rename(columns = {'DeletedPaperIds':'PaperIds'})
    validset = pd.read_csv('dataRev2/Valid.csv')

    print("Getting features for deleted papers from the database")
    features_conf = get_features(dataset, train_confirmed)

    print("Getting features for confirmed papers from the database")
    features_deleted = get_features(dataset, train_deleted)

    print("Getting features for valid papers from the database")
    features_valid = get_features(dataset, validset)

    pickle.dump(features_deleted, open(data_io.get_paths()["deleted_features"], 'wb'))
    pickle.dump(features_conf, open(data_io.get_paths()["confirmed_features"], 'wb'))
    pickle.dump(features_valid, open(data_io.get_paths()["valid_features"], 'wb'))

if __name__=="__main__":
    main()
