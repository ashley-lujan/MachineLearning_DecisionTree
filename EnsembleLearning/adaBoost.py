from weightedSet import WeightedX, Classifiers
from weightedDecAlgo import DecisionTree

import statistics

#returns a list of weightedX's
def saveDataSet(filename):
    result = []
    with open(filename, 'r') as f:
        for line in f:
            terms = line.strip().split(',')
            row = []
            for el in terms:
                row.append(el)
            result.append(row)
    return result

#returns a list of weightedX's
def saveWeightedSet(filename):
    result = []
    with open(filename, 'r') as f:
        for line in f:
            terms = line.strip().split(',')
            row = []
            for el in terms:
                row.append(el)
            result.append(WeightedX(1, row))
    return result

def processBankDataSet(dataSet, attrs): 
    medians = {}
    for i, attr in enumerate(attrs):
            if attrs[attr] == "numeric":
                med = findMedian(dataSet, i)
                medians[i] = med
                replaceWithBinary(dataSet, i, med)
                attrs[attr] = [DecisionTree.less, DecisionTree.greaterE]
    return medians

def findMedian(dataSet, i):
        l = []
        for row in dataSet:
            l.append(int(row.data[i]))
        return statistics.median(l)

def replaceWithBinary(dataSet, i, med):
    for row in dataSet:
        val = int(row.data[i])
        if val >= med:
            row.data[i] = DecisionTree.greaterE
        else:
            row.data[i] = DecisionTree.less

def evaluateModel(d3, csv):
    testSet = saveDataSet('../datasets/bank/' + csv + '.csv')
    correct = 0
    for test in testSet:
        prediction = d3.predict(d3.root, test)
        if prediction == test[len(test) - 1]:
            correct += 1
    return correct/len(testSet)

def getBankAttributes() :
    bank_attributes = {
        "age": "numeric",
        "job": ["admin.", "unknown", "unemployed", "management", "housemaid", "entrepreneur", "student", "blue-collar", "self-employed", "retired", "technician", "services"],
        "marital": ["married", "divorced", "single"],
        "education": ["unknown", "secondary", "primary", "tertiary"],
        "default": ["yes", "no"],
        "balance": "numeric",
        "housing": ["yes", "no"],
        "loan": ["yes", "no"],
        "contact": ["unknown", "telephone", "cellular"],
        "day": "numeric",
        "month": ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
        "duration": "numeric",
        "campaign": "numeric",
        "pdays": "numeric",
        "previous": "numeric",
        "poutcome": ["unknown", "other", "failure", "success"]
    }
    return bank_attributes

def adaBoost():
    dataSet = saveWeightedSet('datasets/bank/train.csv')
    attributes = getBankAttributes()
    dataMedians = processBankDataSet(dataSet=dataSet, attrs=attributes)
    #list of Classifiers
    #Decision Tree should update the weights from dataset and just pass it to other iterations
    trees = []
    T = 1
    for t in range(T): 
        d3 = DecisionTree(dataSet=dataSet, attributesWithValues=attributes, dataMedians=dataMedians, max_depth=2, replaceMissing=True)
        print("uh")
        d3.printTree(d3.root, "")
        # print(evaluateModel(d3, "test"))
        # trees.append(Classifiers(d3, d3.at))
        print("et: ", d3.at)

if __name__ == '__main__':
    adaBoost()