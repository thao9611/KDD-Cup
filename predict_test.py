from predict import *

def main():
    data = pickle.load(open(data_io.get_paths()["test_features"], 'rb'))
    predict_write(data)

if __name__=="__main__":
    main()