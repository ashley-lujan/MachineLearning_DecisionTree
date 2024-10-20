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
        index = randint(0, size-1)
        element_i = lst[index]
        randomized_lst.append(element_i)
    return randomized_lst

def adaBoost():
    trainSet = saveDataSet('../datasets/bank/train.csv')
    testSet = saveDataSet('../datasets/bank/test.csv')
    dataSet = saveWeightedSet('../datasets/bank/train.csv')
    attributes = getBankAttributes()
    # attributes = simple
    dataMedians = {}
    dataMedians = processBankDataSet(dataSet=dataSet, attrs=attributes)
    #list of Classifiers
    #Decision Tree should update the weights from dataset and just pass it to other iterations
    trees = []
    adaBoostErrors = {
        TRAIN : [],
        TEST : []
    }

    stumpErrors = {
        TRAIN : [],
        TEST : []
    }

    T = 20
    for t in range(T): 
        d3 = WeightedDecisionTree(dataSet=dataSet, attributesWithValues=attributes, dataMedians=dataMedians, max_depth=2, replaceMissing=True)
        d3.reweigh_dataSet(dataSet)
        trees.append(d3)

        trainAdaError = evaluateAdaBoost(trainSet, trees)
        testAdaError = evaluateAdaBoost(testSet, trees)

        adaBoostErrors[TRAIN].append(trainAdaError)
        adaBoostErrors[TEST].append(testAdaError)

        trainStumpError = evaluateModel(d3, trainSet)
        testStumpError = evaluateModel(d3, testSet)

        stumpErrors[TRAIN].append(trainStumpError)
        stumpErrors[TEST].append(testStumpError)

    #he first figure shows how the training and test errors vary along with T. The second figure shows the training
    # and test errors of all the decision stumps learned in each iteration.

    displayPlot("AdaBoost Error for each additional Tree", T,"Number of Trees", "Error", adaBoostErrors[TRAIN], adaBoostErrors[TEST])
    displayPlot("Stump Error for each  tree", T,"Tree", "Error", stumpErrors[TRAIN], stumpErrors[TEST])

def displayPlot(title, T, x_title, y_title, y1, y2):
    x = np.arange(0, T, 1)
    plt.plot(x, y1, label='Training Data')
    plt.plot(x, y2, label='Testing Data')

    plt.xlabel(x_title)
    plt.ylabel(y_title)

    plt.legend()

    # Add a title
    plt.title(title)
    plt.show()


def baggedTrees(): 
    dataSet = saveDataSet('../datasets/bank/train.csv')
    testSet = saveDataSet('../datasets/bank/test.csv')
    trees = []
    bagging_errors = {
        TRAIN : [],
        TEST : []
    }
    tree_errors = {
        TRAIN: [],
        TEST: []
    }
    T = 500
    m = len(dataSet)
    print("creating", T, " trees")
    for t in range(T): 
        baggedSet = randomizeSet(dataSet, m)
        d3 = DecisionTree(dataSet=baggedSet, attributesWithValues=getBankAttributes(), hasNumerics=True, max_depth= math.inf , replaceMissing=True)
        trees.append(d3)

        bagging_errors[TRAIN].append( evaluateBagging(dataSet, trees))
        bagging_errors[TEST].append( evaluateBagging(testSet, trees))

        tree_errors[TRAIN].append( evaluateModel(d3, dataSet) )
        tree_errors[TEST].append( evaluateModel(d3, testSet) )
        if t % 20 == 0:
            print("at", t, "tree")

    print("starting plot")

    x = np.arange(0, T, 1)
    plt.plot(x, bagging_errors[TRAIN], label='Training Data for Bagging')
    plt.plot(x, bagging_errors[TEST], label='Testing Data for Bagging')

    plt.plot(x, tree_errors[TRAIN], label='Training Data for Individual')
    plt.plot(x, tree_errors[TEST], label='Testing Data for Individual')

    plt.xlabel("Trees")
    plt.ylabel("Accuracy")

    plt.legend()

    # Add a title
    plt.title("Bagging Implementation Accuracy Across Trees")
    plt.show()

    #Vary the number of trees from 1 to 500, report how the training and test errors vary along with the
    # tree number in a figure.

def experiment(): 
    experiment_runs = 100
    training_size = 1000
    trees_to_learn = 500
    first_predictor_size = 100

    dataSet = saveDataSet('../datasets/bank/train.csv')
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
    dataSet = saveDataSet('../datasets/bank/train.csv')
    testSet = saveDataSet('../datasets/bank/test.csv')
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

    x = np.arange(0, int(50/3) + 1, 1)
    y1 = evaluations[2][TRAIN]
    y2 = evaluations[2][TEST]

    plt.plot(x, y1, label='Training Data')
    plt.plot(x, y2, label='Testing Data')

    # Add labels for the axes
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    # Add a title
    plt.title('Model Performance Over Epochs')

    # Show the legend to differentiate between the lines
    plt.legend()

    # Display the plot
    plt.show()


    
    



        






if __name__ == '__main__':
    adaBoost()
    # baggedTrees()
    # randomForest()
    # Sample data

    # df = px.data.iris()  # Example dataset
    # fig = px.scatter(df, x='sepal_width', y='sepal_length', color='species')
    # fig.show()

    # print(rcsetup.all_backends)
    
    # x = np.arange(0, 10, 0.1)
    # y1 = np.sin(x)
    # y2 = np.cos(x)
    #
    # plt.plot(x, y1, x, y2)
    #
    # plt.show()
