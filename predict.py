# Based on PythonBenchmark code provided offically from KDD13 competition.

from collections import defaultdict
import data_io
import numpy as np
import pickle
import pandas as pd
from extract_feature import parse_paper_ids
from collections import Counter, defaultdict


# Referred https://stackoverflow.com/questions/11236006/identify-duplicate-values-in-a-list-in-python to check duplicate
def processDuplicates(targetList):
    D = defaultdict(list)
    for i, item in enumerate(targetList):
        D[item].append(i)
    D = {k: v for k, v in D.items() if len(v) > 1}

    deleteTargetIdxs = []
    for key, items in D.items():
        sortedItems = sorted(items)
        for i in range(len(sortedItems)):
            if (i == 0):
                continue
            targetList.append(key)
            deleteTargetIdxs.append(sortedItems[i])

    deleteTargetIdxs = sorted(deleteTargetIdxs, reverse = True)

    for i in range(len(deleteTargetIdxs)):
        del targetList[ deleteTargetIdxs[i] ]

    return targetList

def parse_targetset_maintain_duplicate(targetset):
    pair_list = []
    author_id_list = targetset['AuthorId']

    for i in range(len(author_id_list)):
        author_id = author_id_list[i]
        papers = targetset[targetset.AuthorId == author_id]['PaperIds'].unique()[0]
        papers = parse_paper_ids(papers)
        for j in range(len(papers)):
            paper_id = int(papers[j])
            pair_list.append( (author_id, paper_id) )
    return Counter(pair_list)

def predict_write(data, predict_type):
    author_paper_ids = [x[:2] for x in data]
    features = [x[2:] for x in data]

    print("Loading the classifier")
    classifier = data_io.load_model()

    print("Making predictions")

    features = np.array(features)  # This line is for xgboost
    predictions = classifier.predict_proba(features)[:, 1]
    predictions = list(predictions)

    author_predictions = defaultdict(list)
    paper_predictions = {}

    if (predict_type == "valid"):
        targetset = pd.read_csv('dataRev2/Valid.csv')
    else:
        targetset = pd.read_csv('dataRev2/Test.csv')

    parsed_counter = parse_targetset_maintain_duplicate(targetset)

    for (a_id, p_id), pred in zip(author_paper_ids, predictions):
        author_predictions[a_id].append((pred, p_id))

    for author_id in sorted(author_predictions):

        paper_ids_sorted = sorted(author_predictions[author_id], reverse=True)

        new_result = []
        for x in paper_ids_sorted:
            pid = x[1]
            for i in range(parsed_counter[author_id, pid]):
                new_result.append(pid)

        paper_predictions[author_id] = new_result
        #paper_predictions[author_id] = [x[1] for x in paper_ids_sorted]


        paper_predictions[author_id] = processDuplicates(paper_predictions[author_id])

    print("Writing predictions to file")
    data_io.write_submission(paper_predictions, predict_type)

def main():
    data = pickle.load(open(data_io.get_paths()["valid_features"], 'rb'))
    predict_write(data, "valid")

if __name__=="__main__":
    main()