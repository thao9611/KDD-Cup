import pandas as pd
from collections import Counter, defaultdict
import Levenshtein

# functions should return the following format (dictionary)
# { (a1, p1): feature_value1, (a2, p2): feature_value2 ... }

def get_paper_has_how_many_author_in_PaperAuthor(dataset, author_paper_pairs):
    feature_dict = {}

    pid_to_count = dataset['paper_author']['PaperId'].value_counts()
    for ap_pair in author_paper_pairs:
        feature_dict[ap_pair] = pid_to_count[ap_pair[1]]

    return feature_dict

def get_author_publishes_how_many_paper_in_PaperAuthor(dataset, author_paper_pairs):
    feature_dict = {}

    aid_to_count = dataset['paper_author']['AuthorId'].value_counts()
    for ap_pair in author_paper_pairs:
        feature_dict[ap_pair] = aid_to_count[ap_pair[0]]

    return feature_dict

def get_author_publishes_on_how_many_papers_in_conference_of_target_paper_in_PaperAuthor(dataset, author_paper_pairs):
    feature_dict = {}

    papers = dataset['paper'].set_index('Id')
    ac_counts = dataset['ac_count']

    for ap_pair in author_paper_pairs:
        aid = ap_pair[0]
        pid = ap_pair[1]
        cid = papers.loc[pid]['ConferenceId']

        #If id is 0, it means that it has no informaiton about conference or journal.
        count = 0
        if (cid != 0):
            count = ac_counts.loc[aid, float(cid)].iloc[0]
            assert (count > 0), "counting error"

        feature_dict[ap_pair] = count
    return feature_dict

def get_author_publishes_on_how_many_papers_in_journal_of_target_paper_in_PaperAuthor(dataset, author_paper_pairs):
    feature_dict = {}

    papers = dataset['paper'].set_index('Id')
    aj_counts = dataset['aj_count']

    for ap_pair in author_paper_pairs:
        aid = ap_pair[0]
        pid = ap_pair[1]
        cid = papers.loc[pid]['JournalId']

        #If id is 0, it means that it has no informaiton about conference or journal.
        count = 0
        if (cid != 0):
            count = aj_counts.loc[aid, float(cid)].iloc[0]
            assert (count > 0), "counting error"

        feature_dict[ap_pair] = count
    return feature_dict

def get_how_many_duplicated_coauthors_of_target_paper_in_PaperAuthor(dataset, author_paper_pairs):
    feature_dict = {}

    #ap = dataset['ap_duplicate']
    #pa = ap.swaplevel()
    pa = dataset['pa_duplicate']

    # This feature takes too much time because of lengthy boolean array generation.
    for ap_pair in author_paper_pairs:
        aid = ap_pair[0]
        pid = ap_pair[1]
        author_counts = pa.loc[pid]
        count = author_counts[author_counts['Affiliation'] >= 2].count().iloc[0]
        feature_dict[ap_pair] = count
    return feature_dict

# Referred https://stackoverflow.com/questions/11236006/identify-duplicate-values-in-a-list-in-python to check duplicate
def get_duplicated_papers(targetList):
    duplicates = [k for k, v in Counter(targetList).items() if v > 1]
    return duplicates

def make_duplicates_from_targets(alltargets):
    feature_dict = {}

    targetsets = alltargets.sort_values(['AuthorId']).reset_index(drop=True)
    for i in range(len(targetsets)):
        aid = targetsets['AuthorId'][i]
        papers = targetsets['PaperIds'][i].strip().split()
        dups = get_duplicated_papers(papers)
        for pid in dups:
            feature_dict[(aid, int(pid))] = 1
    return feature_dict

def get_how_many_duplicated_papers_of_target_author_in_targetsets(dataset, author_paper_pairs):
    feature_dict = {}
    all_dups = dataset['all_duplicates']
    keys = all_dups.keys()

    for ap_pair in author_paper_pairs:
        if (ap_pair in keys):
            feature_dict[ap_pair] = all_dups[ap_pair]
        else:
            feature_dict[ap_pair] = 0
    return feature_dict

# Consider latin characters, initials, missing name parts
#Amelie M. Achim
#Amélie Achim

# Consider names in ()
#Cefe Lopez
#Cefe Lopez (Cefe López)
def compare_name_similarity(info1, info2):
    #print("i1: ", info1)
    #print("i2: ", info2)
    cont1 = info1.split()
    cont1.sort()
    cont2 = info2.split()
    cont2.sort()

    #print(cont1)
    #print(cont2)
    sorted_info1 = "".join(cont1)
    sorted_info2 = "".join(cont2)

    for i in range(len(cont1)):
        cont1[i] = cont1[i][0]
    for i in range(len(cont2)):
        cont2[i] = cont2[i][0]

    new1 = "".join(cont1)
    new2 = "".join(cont2)

    diff = Levenshtein.distance(new1, new2)

    if (diff < 2):
        return True
    else:
        commons = LCS(sorted_info1, sorted_info2)
        if (commons >= 5):
            return True
    return False


# Get Longest Common Subsequent algorithm from https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Longest_common_subsequence#Computing_the_length_table_of_the_LCS
def LCS(X, Y):
    m = len(X)
    n = len(Y)
    # An (m+1) times (n+1) matrix
    C = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            if X[i-1] == Y[j-1]:
                C[i][j] = C[i-1][j-1] + 1
            else:
                C[i][j] = max(C[i][j-1], C[i-1][j])
    return C[m][n]

def compare_author_name_from_profile(dataset, author_paper_pairs):
    feature_dict = {}

    authors = dataset['author'].set_index('Id')
    authors_index = authors.index
    paper_authors = dataset['ap_indexed']

    for ap_pair in author_paper_pairs:
        aid, pid = ap_pair
        target_name = paper_authors.loc[ap_pair].iloc[0]['Name'].strip()
        if (aid not in authors_index):
            feature_dict[ap_pair] = 7
        else:
            origin_name = authors.loc[aid]['Name'].strip()
            if (origin_name == "" or target_name == ""):
                feature_dict[ap_pair] = 7
            else:
                feature_dict[ap_pair] = Levenshtein.distance(authors.loc[aid]['Name'], target_name)
                if (feature_dict[ap_pair] > 0):
                    if (compare_name_similarity(origin_name, target_name)):
                        feature_dict[ap_pair] = 0
                    else:
                        #print(ap_pair, feature_dict[ap_pair])
                        #print(authors.loc[aid]['Name'])
                        #print(paper_authors.loc[ap_pair].iloc[0]['Name'])
                        pass

    return feature_dict


# split by | to take care of below example
# National Taiwan University
# Dept . of Electrical Engineering and Graduate|Institute of Communication Engineering|National Taiwan University

def compare_affiliaton_similarity_levenshtein(info1, info2):
    cont1 = []
    cont2 = []
    if ("|" in info1):
        cont1 = info1.strip().split("|")
    else:
        cont1.append(info1)

    if ("|" in info2):
        cont2 = info2.strip().split("|")
    else:
        cont2.append(info2)

    value_min = 100000
    for c1 in cont1:
        for c2 in cont2:
            dist = Levenshtein.distance(c1, c2)
            value_min = min(value_min, dist)

    return value_min


def compare_author_affiliation_from_profile(dataset, author_paper_pairs):
    feature_dict = {}

    authors = dataset['author'].set_index('Id')
    authors_index = authors.index
    paper_authors = dataset['ap_indexed']

    for ap_pair in author_paper_pairs:
        aid, pid = ap_pair
        target_name = paper_authors.loc[ap_pair].iloc[0]['Affiliation'].strip()
        if (aid not in authors_index):
            feature_dict[ap_pair] = 10
        else:
            origin_name = authors.loc[aid]['Affiliation'].strip()
            if (origin_name == "" or target_name == ""):
                # If both are empty, more likely to be rejected
                if (origin_name == "" and target_name == ""):
                    feature_dict[ap_pair] = 13
                else:
                    feature_dict[ap_pair] = 5
                #print("ori: ", origin_name)
                #print("trg: ", target_name)
            else:
                feature_dict[ap_pair] = compare_affiliaton_similarity_levenshtein(authors.loc[aid]['Affiliation'], target_name)
                #print(ap_pair, feature_dict[ap_pair])
                #print(authors.loc[aid]['Affiliation'])
                #print(paper_authors.loc[ap_pair].iloc[0]['Affiliation'])
            assert feature_dict[ap_pair] < 100000, "Too big value"

    return feature_dict