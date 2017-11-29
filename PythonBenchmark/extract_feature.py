import pickle
import data_io

def main():
    print("Getting features for deleted papers from the database")
    features_deleted = data_io.get_features_db("TrainDeleted")
    print(features_deleted)

    print("Getting features for confirmed papers from the database")
    features_conf = data_io.get_features_db("TrainConfirmed")

    print("Getting features for valid papers from the database")
    data = data_io.get_features_db("ValidPaper")

    pickle.dump(features_deleted, open(data_io.get_paths()["deleted_features"], 'w'))
    pickle.dump(features_conf, open(data_io.get_paths()["confirmed_features"], 'w'))
    pickle.dump(data, open(data_io.get_paths()["valid_features"], 'w'))

if __name__=="__main__":
    main()