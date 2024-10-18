from weightedSet import WeightedX, Classifiers
from weightedDecAlgo import WeightedDecisionTree
from decAlgo import DecisionTree
from randomForest import RandomForestTree
from random import randint
import statistics, math

import matplotlib.pyplot as plt
import numpy as np



TRAIN = "train"
TEST = "test"

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
    
    m = len(result)
    for r in result: 
        r.weight = 1/m
    return result

def processBankDataSet(dataSet, attrs): 
    medians = {}
    for i, attr in enumerate(attrs):
            if attrs[attr] == "numeric":
                med = findMedian(dataSet, i)
                medians[i] = med
                replaceWithBinary(dataSet, i, med)
                attrs[attr] = [WeightedDecisionTree.less, WeightedDecisionTree.greaterE]
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
            row.data[i] = WeightedDecisionTree.greaterE
        else:
            row.data[i] = WeightedDecisionTree.less

def evaluateModel(d3, testSet):
    correct = 0
    for test in testSet:
        prediction = d3.predict(test)
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

simple = {
    "a": [0, 1, 2],
    "b": [0, 1, 2],
    "c": [0, 1, 2]
}

def evaluateAdaBoost(testSet, trees): 
    correct = 0
    for test in testSet:
        prediction = calcFinal(trees, test)
        if prediction == test[len(test) - 1]:
            correct += 1
    return correct/len(testSet)

def evaluateBagging(testSet, trees): 
    correct = 0
    for test in testSet:
        prediction = calcFinalBoosting(trees, test)
        if prediction == test[len(test) - 1]:
            correct += 1
    return correct/len(testSet)

def calcFinal(trees, x): 
    sum = 0 
    for tree in trees: 
        sum += tree.at * tree.translate(tree.predict(x))
    if sum > 0: 
        return "yes"
    return "no"

def calcFinalBoosting(trees, x): 
    sum = 0 
    length = len(trees)
    for tree in trees: 
        sum += ((1/length) * tree.translate(tree.predict(x)))
    if sum > 0: 
        return "yes"
    return "no"


def randomizeSet(lst, size): 
    randomized_lst = []
    for i in range(size): 
        index = randint(0, m-1)
        element_i = lst[index]
        randomized_lst.append(element_i)
    return randomized_lst

def adaBoost():
    trainSet = saveDataSet('datasets/bank/train.csv')
    testSet = saveDataSet('datasets/bank/test.csv')
    dataSet = saveWeightedSet('datasets/bank/train.csv')
    attributes = getBankAttributes()
    # attributes = simple
    dataMedians = {}
    dataMedians = processBankDataSet(dataSet=dataSet, attrs=attributes)
    #list of Classifiers
    #Decision Tree should update the weights from dataset and just pass it to other iterations
    trees = []
    T = 20
    for t in range(T): 
        d3 = WeightedDecisionTree(dataSet=dataSet, attributesWithValues=attributes, dataMedians=dataMedians, max_depth=2, replaceMissing=True)
        d3.reweigh_dataSet(dataSet)
        trees.append(d3)
        print(evaluateAdaBoost(trainSet, trees))

def baggedTrees(): 
    dataSet = saveDataSet('datasets/bank/train.csv')
    testSet = saveDataSet('datasets/bank/test.csv')
    print("hello")
    trees = []
    T = 100
    m = len(lst)
    for t in range(T): 
        baggedSet = randomizeSet(dataSet, m)
        d3 = DecisionTree(dataSet=baggedSet, attributesWithValues=getBankAttributes(), hasNumerics=True, max_depth= math.inf , replaceMissing=True)
        trees.append(d3)
        # print(evaluateBagging(dataSet, trees))

def experiment(): 
    experiment_runs = 100
    training_size = 1000
    trees_to_learn = 500
    first_predictor_size = 100

    dataSet = saveDataSet('datasets/bank/train.csv')
    predictors = []
    for run in range(experiment_runs): 
        baggedSet = randomizeSet(dataSet, training_size)
        trees = []
        #todo: I don't know what the heck this means
        #For comparison, pick the first tree in each run to get 100 fully expanded trees (i.e. single trees)
        #the rest get full 
        for tree_to_learn in range(trees_to_learn): 
            d3 = DecisionTree(dataSet=baggedSet, attributesWithValues=getBankAttributes(), hasNumerics=True, max_depth= math.inf , replaceMissing=True)
            trees.append(d3)
        predictors.append(trees)

def randomForest(): 
    dataSet = saveDataSet('datasets/bank/train.csv')
    testSet = saveDataSet('datasets/bank/test.csv')
    evaluations = {
        2 : {
            TRAIN: [], 
            TEST: []
        }, 
        4 : {
            TRAIN: [], 
            TEST: []
        }, 
        6 : {
            TRAIN: [], 
            TEST: []
        }
    }
    T = 50
    trees = []
    for t in range(T): 
        random_set_size = (t % 3) * 2 + 2
        d3 = RandomForestTree(random_set_size, dataSet=dataSet, attributesWithValues=getBankAttributes(), hasNumerics=True, max_depth= math.inf , replaceMissing=True)
        trees.append(d3)
        train_error = evaluateModel(d3, dataSet)
        test_error = evaluateModel(d3, testSet)
        evaluations[random_set_size][TRAIN].append(train_error)
        evaluations[random_set_size][TEST].append(test_error)
    
    for i in range(2, 8, 2): 
        print("with size", i)
        question = evaluations[i]
        for j in question[TEST]:
            print(j)


    
    



        



import matplotlib.rcsetup as rcsetup
import plotly.express as px



if __name__ == '__main__':
    # adaBoost()
    # baggedTrees()
    # randomForest()
    # Sample data

    df = px.data.iris()  # Example dataset
    fig = px.scatter(df, x='sepal_width', y='sepal_length', color='species')
    fig.show()

    # print(rcsetup.all_backends)
    
    # x = np.arange(0, 10, 0.1)
    # y1 = np.sin(x)
    # y2 = np.cos(x)

    # plt.plot(x, y1, x, y2)

    # plt.show()

    while True: 
        l = input()
        if l == "quit": 
            break