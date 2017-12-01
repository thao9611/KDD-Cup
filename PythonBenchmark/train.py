import data_io
import xgboost as xgb
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot
import pickle
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search

def main():
    '''
    print("Getting features for deleted papers from the database")
    features_deleted = data_io.get_features_db("TrainDeleted")

    print("Getting features for confirmed papers from the database")
    features_conf = data_io.get_features_db("TrainConfirmed")
    '''

    features_deleted = pickle.load(open(data_io.get_paths()["deleted_features"]))
    features_conf = pickle.load(open(data_io.get_paths()["confirmed_features"]))

    features = [x[2:] for x in features_deleted + features_conf]
    target = [0 for x in range(len(features_deleted))] + [1 for x in range(len(features_conf))]

    print("Training the Classifier")

    features = np.array(features)
    target = np.array(target)
    '''
    classifier = RandomForestClassifier(n_estimators=50, 
                                        verbose=2,
                                        n_jobs=1,
                                        min_samples_split=10,
                                        random_state=1)
    classifier.fit(features, target)
    '''

    #Referred https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/ for parameter tuning

    '''
    param_test1 = {
        'max_depth': range(3, 10, 1),
        'min_child_weight': range(1, 6, 1)
    }
    gsearch1 = GridSearchCV(estimator=xgb.XGBClassifier(learning_rate=0.1, n_estimators=140, max_depth=5,
                                                    min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                                    objective='binary:logistic', nthread=4, scale_pos_weight=1,
                                                    seed=27),
                            param_grid=param_test1, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
    gsearch1.fit(features, target)
    print(gsearch1.grid_scores_)
    print(gsearch1.best_params_)
    print(gsearch1.best_score_)
    exit()
    '''

    '''
    classifier = xgb.XGBClassifier(learning_rate=0.1, n_estimators=140, max_depth=7,
                                                    min_child_weight=3, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                                    objective='binary:logistic', nthread=4, scale_pos_weight=1,
                                                    seed=27).fit(features, target)
    '''
    classifier = xgb.XGBClassifier(max_depth=8, n_estimators=300, learning_rate=0.05, objective="binary:logistic").fit(features, target)

    # plot
    xgb.plot_importance(classifier)
    pyplot.show()

    print("Saving the classifier")
    data_io.save_model(classifier)
    
if __name__=="__main__":
    main()
