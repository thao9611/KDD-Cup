import pickle
import data_io
import pandas as pd
import numpy as np
from feature_set import *
from Thao_features import *
from kamil_features import *
import time
import unidecode

# Referred below for latin character to english character
# https://stackoverflow.com/questions/517923/what-is-the-best-way-to-remove-accents-in-a-python-unicode-string
def read_all_DS():
    dataset = {}
    dataset['paper'] = pd.read_csv('dataRev2/Paper.csv')
    dataset['author'] = pd.read_csv('dataRev2/Author.csv')
    dataset['conference'] = pd.read_csv('dataRev2/Conference.csv')
    dataset['journal'] = pd.read_csv('dataRev2/Journal.csv')
    dataset['paper_author'] = pd.read_csv('dataRev2/PaperAuthor.csv')
    #dataset['tokens'] = pickle.load(open(data_io.get_paths()["paper_title_tokens"], 'rb'))

    dataset['author']['Name'].fillna("", inplace=True)
    dataset['author']['Name'] = dataset['author']['Name'].str.lower().apply(unidecode.unidecode)

    dataset['author']['Affiliation'].fillna("", inplace=True)
    dataset['author']['Affiliation'] = dataset['author']['Affiliation'].str.lower().apply(unidecode.unidecode)

    new_pa = dataset['paper_author']
    new_pa['Name'].fillna("", inplace=True)
    new_pa['Name'] = new_pa['Name'].str.lower().apply(unidecode.unidecode)

    new_pa['Affiliation'].fillna("", inplace=True)
    new_pa['Affiliation'] = new_pa['Affiliation'].str.lower().apply(unidecode.unidecode)


    merged_info = pd.merge(dataset['paper_author'], dataset['paper'], how='left', left_on='PaperId', right_on='Id')
    dataset['ac_count'] = merged_info[['AuthorId', 'PaperId','ConferenceId']]\
        .groupby(['AuthorId', 'ConferenceId']).size().reset_index(name='counts').set_index(['AuthorId', 'ConferenceId']).sort_index()
    dataset['aj_count'] = merged_info[['AuthorId', 'PaperId','JournalId']]\
        .groupby(['AuthorId', 'JournalId']).size().reset_index(name='counts').set_index(['AuthorId', 'JournalId']).sort_index()

    dataset['ap_duplicate'] = pd.DataFrame(pd.pivot_table(new_pa, values = "Affiliation",index = ['AuthorId',"PaperId"], aggfunc = "count")).sort_index()
    dataset['pa_duplicate'] = pd.DataFrame(
        pd.pivot_table(new_pa, values="Affiliation", index=['PaperId', "AuthorId"], aggfunc="count")).sort_index()

    dataset['ap_indexed'] = new_pa.set_index(['AuthorId','PaperId']).sort_index()

    '''
    start_time = time.time()
    pickle.dump(dataset, open("results/dataset.pickle", 'wb'))
    print("dump: ", time.time() - start_time)
    dataset = pickle.load(open("results/dataset.pickle", 'rb'))
    print ("dataset load done!")
    '''

    return dataset

def parse_paper_ids(paper_ids_string):
    return paper_ids_string.strip().split()

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
    harry_list = []
    thao_list = []
    kamil_list = []

    # Keep the format of f# (dictionary): { (a1, p1): feature_value1, (a2, p2): feature_value2 ... }
    # Add your features here and add them to feature_list!

    harry_f1 = get_author_publishes_how_many_paper_in_PaperAuthor(dataset, author_paper_pairs)
    harry_f2 = get_paper_has_how_many_author_in_PaperAuthor(dataset, author_paper_pairs)

    harry_list = [harry_f1, harry_f2] # Default features


    harry_f3 = get_author_publishes_on_how_many_papers_in_conference_of_target_paper_in_PaperAuthor(dataset, author_paper_pairs)
    harry_f4 = get_author_publishes_on_how_many_papers_in_journal_of_target_paper_in_PaperAuthor(dataset, author_paper_pairs)
    harry_f5 = get_how_many_duplicated_coauthors_of_target_paper_in_PaperAuthor(dataset, author_paper_pairs)
    harry_f6 = get_how_many_duplicated_papers_of_target_author_in_targetsets(dataset, author_paper_pairs)
    thao_f1 = author_paper_frequency_count(dataset, author_paper_pairs)
    # thao_f3 = target_paper_and_papers_of_target_author_by_keywords(dataset, author_paper_pairs)
    harry_list += [harry_f3, harry_f4, thao_f1, harry_f5, harry_f6] # 97.06% accuracy


    harry_f7 = compare_author_name_from_profile(dataset, author_paper_pairs)
    harry_f8 = compare_author_affiliation_from_profile(dataset, author_paper_pairs)
    harry_list += [harry_f7, harry_f8]
    feature_list = harry_list

    '''
    start_time = time.time()
    print(start_time)
    kamil_f1 = kamil_feature_11(dataset, author_paper_pairs)
    print(time.time() - start_time)

    kamil_list = [kamil_f1]
    feature_list = kamil_list + harry_list

    #thao_f1 = author_paper_frequency_count(dataset, author_paper_pairs)
    #thao_f2 = author_paper_affiliation(dataset, author_paper_pairs)
    #thao_f3 = target_paper_and_papers_of_target_author_by_keywords(dataset,author_paper_pairs)
  
    #thao_f4 = target_paper_and_papers_of_target_author_by_years(dataset, author_paper_pairs)

    #thao_list =[thao_f1, thao_f3,thao_f4]
    #feature_list = harry_list + kamil_list + thao_list
    '''

    result_list = generate_feature_list(author_paper_pairs, feature_list)
    return result_list

def main():
    print("Reading csv files")
    dataset = read_all_DS()

    trainset = pd.read_csv('dataRev2/Train.csv')
    train_confirmed = trainset[['AuthorId', 'ConfirmedPaperIds']].rename(columns = {'ConfirmedPaperIds':'PaperIds'})
    train_deleted = trainset[['AuthorId', 'DeletedPaperIds']].rename(columns = {'DeletedPaperIds':'PaperIds'})
    validset = pd.read_csv('dataRev2/Valid.csv')
    testset = pd.read_csv('dataRev2/Test.csv')

    allsets = pd.concat([train_confirmed, validset, testset])
    all_dups = make_duplicates_from_targets(allsets)
    dataset['all_duplicates'] = all_dups


    print("Getting features for confirmed papers")
    features_conf = get_features(dataset, train_confirmed)

    print("Getting features for deleted papers")
    features_deleted = get_features(dataset, train_deleted)

    print("Getting features for valid papers")
    features_valid = get_features(dataset, validset)

    pickle.dump(features_deleted, open(data_io.get_paths()["deleted_features"], 'wb'))
    pickle.dump(features_conf, open(data_io.get_paths()["confirmed_features"], 'wb'))
    pickle.dump(features_valid, open(data_io.get_paths()["valid_features"], 'wb'))

    '''
    print("Getting features for test papers")
    features_test = get_features(dataset, testset)
    pickle.dump(features_test, open(data_io.get_paths()["test_features"], 'wb'))
    '''

if __name__=="__main__":
    main()
