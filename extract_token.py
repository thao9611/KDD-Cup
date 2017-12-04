from Thao_features import *

def main():
    print("Extracting keywords from paper titles")
    dataset = {}
    dataset['paper'] = pd.read_csv('dataRev2/Paper.csv')
    paper_keywords(dataset)

if __name__=="__main__":
    main()

