import pickle

def merge_features(path1, path2, result_path, target_files):
    for filename in target_files:
        result = []
        feature1 = pickle.load(open(path1 + filename, 'rb'))
        feature2 = pickle.load(open(path2 + filename, 'rb'))

        assert len(feature1) == len(feature2), "Total instance length not matched."
        for i in range(len(feature1)):
            f1 = feature1[i]
            f2 = feature2[i]

            assert f1[0:2] == f2[0:2], "Feature id not matced."

            f2 = f2[4:] # TODO: change when there is overlapping feature
            f1 += f2
            result.append(f1)

        pickle.dump(result, open(result_path + filename, 'wb'))




def main():
    path1 = 'results/feature_group1_9706/'
    path2 = 'results/feature_group2_8434/'
    result_path = 'results/feature_merged/'
    target_files = ["confirmedFeatures.pickle"]
    target_files += ["deletedFeatures.pickle", "validFeatures.pickle"]
    merge_features(path1, path2, result_path, target_files)

if __name__=="__main__":
    main()

