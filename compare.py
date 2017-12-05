# Based on PythonBenchmark code provided offically from KDD13 competition.

import pandas as pd
from collections import Counter

# Referred https://stackoverflow.com/questions/11236006/identify-duplicate-values-in-a-list-in-python to check duplicate
def checkDuplicate(targetList, authorId):
    duplicates = [k for k, v in Counter(targetList).items() if v > 1]

    #if ((len(duplicates) > 0)):
    #    print ("Duplicate occurs! AuthorId: %s, PaperIds: %s" % (authorId, duplicates))
    assert (len(duplicates) == 0), "Duplicate occurs! AuthorId: %s, PaperIds: %s" % (authorId, duplicates)


def getAvgPrecision(predictPapers, solutionPapers):
    confirmArray = len(predictPapers) * [False]
    for i in range(len(solutionPapers)):
        targetPaper = solutionPapers[i]
        predIdx = predictPapers.index(targetPaper)
        confirmArray[predIdx] = True

    accTrue = 0
    accFalse = 0
    accPrecision = 0
    for i in range(len(predictPapers)):
        if (confirmArray[i]):
            accTrue += 1
            accPrecision += (accTrue) / (i + 1)
        else:
            accFalse += 1

        assert (accTrue + accFalse == i + 1)
    accPrecision /= len(solutionPapers)
    return accPrecision

if __name__ == '__main__':
    validPredict = pd.read_csv('./results/basicCoauthorBenchmarkRev2.csv')
    validSolution = pd.read_csv('./dataRev2/ValidSolution.csv')

    validPredict = validPredict.sort_values(['AuthorId']).reset_index(drop=True)
    validSolution = validSolution.sort_values(['AuthorId']).reset_index(drop=True)

    assert (len(validPredict) == len(validSolution)), "Length is not matched!"

    #TODO: duplicate check in solution or prediction?

    meanAvgPrecision = 0

    for i in range(len(validPredict)):
        predictAuthorId = validPredict['AuthorId'][i]
        solutionAuthorId = validSolution['AuthorId'][i]

        assert (predictAuthorId == solutionAuthorId), "AuthorId is not equal!"

        predictPapers = validPredict['PaperIds'][i].strip().split()
        solutionPapers = validSolution['PaperIds'][i].strip().split()

        # There are duplicates in predict paper and solution paper.
        #checkDuplicate(predictPapers, predictAuthorId)
        #checkDuplicate(solutionPapers, predictAuthorId)

        assert(getAvgPrecision([7,3,2,4,1,6,5], [7,3]) == getAvgPrecision([7,3,2,4,1,6,5], [3,7]) == 1)
        assert(getAvgPrecision([3,1,4,5,2], [3,5]) == getAvgPrecision([3,1,4,5,2], [5,3]) == 0.75),\
            "Use Python3 to execute this file"
        meanAvgPrecision += getAvgPrecision(predictPapers, solutionPapers)

    meanAvgPrecision /= len(validPredict)
    print(meanAvgPrecision)
