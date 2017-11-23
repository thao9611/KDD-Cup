# Based on PythonBenchmark code from offical KDD13 competition.

import csv
import json
import os
import pickle

def paper_ids_to_string(ids):
    return " ".join([str(x) for x in ids])

def get_paths():
    paths = json.loads(open("SETTINGS.json").read())
    for key in paths:
        paths[key] = os.path.expandvars(paths[key])
    return paths

def save_model(model):
    out_path = get_paths()["model_path"]
    pickle.dump(model, open(out_path, 'wb'))

def load_model():
    in_path = get_paths()["model_path"]
    return pickle.load(open(in_path, 'rb'))

def write_submission(predictions):
    submission_path = get_paths()["submission_path"]
    rows = [(author_id, paper_ids_to_string(predictions[author_id])) for author_id in predictions]
    writer = csv.writer(open(submission_path, "w"), lineterminator="\n")
    writer.writerow(("AuthorId", "PaperIds"))
    writer.writerows(rows)
