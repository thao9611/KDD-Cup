import pandas as pd
from collections import Counter, defaultdict

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



def kamil_new_f1(dataset, author_paper_pairs):
    feature_dict = {}

    #aid_to_count = dataset['paper_author']['AuthorId'].value_counts()
    print(author_paper_pairs[0])
    for ap_pair in author_paper_pairs:
        feature_dict[ap_pair] = int(ap_pair[0]) + int(ap_pair[1])

    return feature_dict