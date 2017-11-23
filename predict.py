from collections import defaultdict
import data_io
import numpy as np
import pickle
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

def main():
    '''
    print("Getting features for valid papers from the database")
    data = data_io.get_features_db("ValidPaper")
    '''
    data = pickle.load(open(data_io.get_paths()["valid_features"], 'rb'))
    author_paper_ids = [x[:2] for x in data]
    features = [x[2:] for x in data]

    print("Loading the classifier")
    classifier = data_io.load_model()

    print("Making predictions")

    features = np.array(features) # This line is for xgboost
    predictions = classifier.predict_proba(features)[:,1]
    predictions = list(predictions)

    author_predictions = defaultdict(list)
    paper_predictions = {}

    for (a_id, p_id), pred in zip(author_paper_ids, predictions):
        author_predictions[a_id].append((pred, p_id))

    for author_id in sorted(author_predictions):
        paper_ids_sorted = sorted(author_predictions[author_id], reverse=True)
        paper_predictions[author_id] = processDuplicates([x[1] for x in paper_ids_sorted])

    print("Writing predictions to file")
    data_io.write_submission(paper_predictions)

if __name__=="__main__":
    main()