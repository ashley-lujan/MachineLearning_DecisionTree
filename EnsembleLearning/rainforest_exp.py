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


# returns a list of weightedX's
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


# returns a list of weightedX's
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
        r.weight = 1 / m
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
    return correct / len(testSet)

class savePredReturn():
    def __init__(self, lst_p, accuracy):
        self.predictions = lst_p
        self.accuracy = accuracy

def savePredictions(d3, testSet):
    predictions = []
    correct = 0
    for test in testSet:
        prediction = d3.predict(test)
        predictions.append(prediction)
        if prediction == test[len(test) - 1]:
            correct += 1
    return savePredReturn(predictions, correct / len(testSet))


def getBankAttributes():
    bank_attributes = {
        "age": "numeric",
        "job": ["admin.", "unknown", "unemployed", "management", "housemaid", "entrepreneur", "student", "blue-collar",
                "self-employed", "retired", "technician", "services"],
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


def evaluateAdaBoost(testSet, trees, saved_pred):
    correct = 0
    for i, test in enumerate(testSet):
        prediction = calcFinalKnowing(trees, saved_pred, test, i)
        if prediction == test[len(test) - 1]:
            correct += 1
    return correct / len(testSet)


def calcFinalKnowing(trees, saved_pred, x, j):
    sum = 0
    for i, tree in enumerate(trees):
        sum += tree.at * translate(saved_pred[i][j])
    if sum > 0:
        return "yes"
    return "no"


def evaluateBagging(testSet, trees):
    correct = 0
    for i, test in enumerate(testSet):
        prediction = calcFinalBoosting(trees, test, i)
        if prediction == test[len(test) - 1]:
            correct += 1
    return correct / len(testSet)


def calcFinal(trees, x, i):
    sum = 0
    for tree in trees:
        sum += tree.at * tree.translate(tree.predict(x))
    if sum > 0:
        return "yes"
    return "no"

def translate(value):
    if value == "yes":
        return 1
    return -1


def calcFinalBoosting(trees, x, i):
    sum = 0
    length = len(trees)
    for tree in trees:
        sum += ((1 / length) * translate(tree[i]))
    if sum > 0:
        return "yes"
    return "no"


def randomizeSet(lst, size):
    randomized_lst = []
    for i in range(size):
        index = randint(0, size - 1)
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
    # list of Classifiers
    # Decision Tree should update the weights from dataset and just pass it to other iterations
    trees = []
    adaBoostErrors = {
        TRAIN: [],
        TEST: []
    }

    stumpErrors = {
        TRAIN: [],
        TEST: []
    }

    trees_train = []
    trees_test = []


    T = 500
    for t in range(T):
        d3 = WeightedDecisionTree(dataSet=dataSet, attributesWithValues=attributes, dataMedians=dataMedians,
                                  max_depth=2, replaceMissing=True)
        d3.reweigh_dataSet(dataSet)
        trees.append(d3)

        #added code
        train_saved = savePredictions(d3, trainSet)
        test_saved = savePredictions(d3, testSet)

        stumpErrors[TRAIN].append(train_saved.accuracy)
        stumpErrors[TEST].append(test_saved.accuracy)

        trees_train.append(train_saved.predictions)
        trees_test.append(test_saved.predictions)

        adaBoostErrors[TRAIN].append(evaluateAdaBoost(trainSet, trees, trees_train))
        adaBoostErrors[TEST].append(evaluateAdaBoost(testSet, trees, trees_test))

        if t % 20 == 0:
            print("at run ", t)

    # he first figure shows how the training and test errors vary along with T. The second figure shows the training
    # and test errors of all the decision stumps learned in each iteration.

    displayPlot("AdaBoost Error for each additional Tree", T, "Number of Trees", "Accuracy", adaBoostErrors[TRAIN],
                adaBoostErrors[TEST])
    displayPlot("Stump Error for each  tree", T, "Tree", "Accuracy", stumpErrors[TRAIN], stumpErrors[TEST])


def displayPlot(title, T, x_title, y_title, y1, y2):
    x = range(0, T, 1)
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

    trees_train = []
    trees_test = []

    bagging_errors = {
        TRAIN: [],
        TEST: []
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
        d3 = DecisionTree(dataSet=baggedSet, attributesWithValues=getBankAttributes(), hasNumerics=True,
                          max_depth=math.inf, replaceMissing=True)
        # trees.append(d3)

        train_saved = savePredictions(d3, dataSet)
        test_saved = savePredictions(d3, testSet)

        tree_errors[TRAIN].append(train_saved.accuracy)
        tree_errors[TEST].append(test_saved.accuracy)

        trees_train.append(train_saved.predictions)
        trees_test.append(test_saved.predictions)

        bagging_errors[TRAIN].append(evaluateBagging(dataSet, trees_train))
        bagging_errors[TEST].append(evaluateBagging(testSet, trees_test))

        # tree_errors[TRAIN].append(evaluateModel(d3, dataSet))
        # tree_errors[TEST].append(evaluateModel(d3, testSet))
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

    # Vary the number of trees from 1 to 500, report how the training and test errors vary along with the
    # tree number in a figure.


def experimentBaggedTrees():
    experiment_runs = 100
    training_size = 1000
    trees_to_learn = 500

    bagged_trees = []
    single_trees = []

    dataSet = saveDataSet('../datasets/bank/train.csv')
    testSet = saveDataSet('../datasets/bank/test.csv')
    predictors = []
    for run in range(experiment_runs):
        baggedSet = randomizeSet(dataSet, training_size)
        single_tree = DecisionTree(dataSet=baggedSet, attributesWithValues=getBankAttributes(), hasNumerics=True,
                              max_depth=math.inf, replaceMissing=True)
        single_trees.append(single_tree)

        trees = []
        # todo: I don't know what the heck this means
        # For comparison, pick the first tree in each run to get 100 fully expanded trees (i.e. single trees)
        # the rest get full
        for tree_to_learn in range(trees_to_learn):
            d3 = DecisionTree(dataSet=baggedSet, attributesWithValues=getBankAttributes(), hasNumerics=True,
                              max_depth=math.inf, replaceMissing=True)
            trees.append(d3)
        predictors.append(trees)
        if run % 20 == 0:
            print("at run ", run)

    #single tree
    sngle_tree_average_bias = 0
    sngle_tree_average_var = 0
    for test in testSet:
        average = 0
        test_predictions = []
        for sngle_tree in single_trees:
            predic = translate(sngle_tree.predict(test))
            test_predictions.append(predic)
            average += predic
        average = average/len(single_trees)
        expected = translate(test[-1])
        bias = math.pow(average - expected, 2)
        sample_var = computeSampleVariance(expected, test_predictions)
        sngle_tree_average_bias += bias
        sngle_tree_average_var += sample_var
    sngle_tree_average_var /= len(single_trees)
    sngle_tree_average_bias /= len(single_trees)
    sngle_squared_error = sngle_tree_average_bias + sngle_tree_average_var
    print("Single Trees")
    print(sngle_tree_average_bias, sngle_tree_average_var, sngle_squared_error)

    #bagged_trees
    # single tree
    sngle_tree_average_bias = 0
    sngle_tree_average_var = 0
    for test in testSet:
        average = 0
        test_predictions = []
        for bag in predictors:
            predic = translate(calcFinal(bag, test, 0))
            test_predictions.append(predic)
            average += predic
        average = average / len(predictors)
        expected = translate(test[-1])
        bias = math.pow(average - expected, 2)
        sample_var = computeSampleVariance(expected, test_predictions)
        sngle_tree_average_bias += bias
        sngle_tree_average_var += sample_var
    sngle_tree_average_var /= len(single_trees)
    sngle_tree_average_bias /= len(single_trees)
    sngle_squared_error = sngle_tree_average_bias + sngle_tree_average_var
    print("Bagged Trees")
    print(sngle_tree_average_bias, sngle_tree_average_var, sngle_squared_error)

def experimentRainForest():
    experiment_runs = 100
    training_size = 1000
    trees_to_learn = 500

    bagged_trees = []
    single_trees = []

    dataSet = saveDataSet('../datasets/bank/train.csv')
    testSet = saveDataSet('../datasets/bank/test.csv')
    predictors = []
    for run in range(experiment_runs):
        # baggedSet = randomizeSet(dataSet, training_size)
        single_tree = RandomForestTree(4, dataSet=dataSet, attributesWithValues=getBankAttributes(), hasNumerics=True,
                              max_depth=math.inf, replaceMissing=True)
        single_trees.append(single_tree)

        trees = []
        # todo: I don't know what the heck this means
        # For comparison, pick the first tree in each run to get 100 fully expanded trees (i.e. single trees)
        # the rest get full
        for tree_to_learn in range(trees_to_learn):
            d3 = RandomForestTree(4, dataSet=dataSet, attributesWithValues=getBankAttributes(), hasNumerics=True,
                              max_depth=math.inf, replaceMissing=True)
            trees.append(d3)
        predictors.append(trees)
        print("at run ", run)

    #single tree
    sngle_tree_average_bias = 0
    sngle_tree_average_var = 0
    for test in testSet:
        average = 0
        test_predictions = []
        for sngle_tree in single_trees:
            predic = translate(sngle_tree.predict(test))
            test_predictions.append(predic)
            average += predic
        average = average/len(single_trees)
        expected = translate(test[-1])
        bias = math.pow(average - expected, 2)
        sample_var = computeSampleVariance(expected, test_predictions)
        sngle_tree_average_bias += bias
        sngle_tree_average_var += sample_var
    sngle_tree_average_var /= len(single_trees)
    sngle_tree_average_bias /= len(single_trees)
    sngle_squared_error = sngle_tree_average_bias + sngle_tree_average_var
    print("Single Trees")
    print(sngle_tree_average_bias, sngle_tree_average_var, sngle_squared_error)

    #bagged_trees
    # single tree
    sngle_tree_average_bias = 0
    sngle_tree_average_var = 0
    for testi, test in enumerate(testSet):
        print("considering rainforest", testi)
        average = 0
        test_predictions = []
        for bag in predictors:
            predic = translate(calcFinal(bag, test, 0))
            test_predictions.append(predic)
            average += predic
        average = average / len(predictors)
        expected = translate(test[-1])
        bias = math.pow(average - expected, 2)
        sample_var = computeSampleVariance(expected, test_predictions)
        sngle_tree_average_bias += bias
        sngle_tree_average_var += sample_var

    sngle_tree_average_var /= len(single_trees)
    sngle_tree_average_bias /= len(single_trees)
    sngle_squared_error = sngle_tree_average_bias + sngle_tree_average_var
    print("Rainforest Trees")
    print(sngle_tree_average_bias, sngle_tree_average_var, sngle_squared_error)

def computeSampleVariance(mu, predictions):
    n = len(predictions)
    sum = 0
    for pred in predictions:
        sum += math.pow(pred - mu, 2)
    result = (sum / (n - 1))
    return result





def randomForest():
    dataSet = saveDataSet('../datasets/bank/train.csv')
    testSet = saveDataSet('../datasets/bank/test.csv')
    T = 500
    for size in range(2, 8, 2):
        get_train_test(dataSet, testSet, T, size)

def get_train_test(dataSet, testSet, T, random_set_size):
    T = 30
    trees_train = []
    trees_test = []
    rain_forst_errors = {
        TRAIN: [],
        TEST: []
    }
    tree_errors = {
        TRAIN: [],
        TEST: []
    }
    for t in range(T):
        d3 = RandomForestTree(random_set_size, dataSet=dataSet, attributesWithValues=getBankAttributes(),
                              hasNumerics=True, max_depth=math.inf, replaceMissing=True)
        train_saved = savePredictions(d3, dataSet)
        test_saved = savePredictions(d3, testSet)

        tree_errors[TRAIN].append(train_saved.accuracy)
        tree_errors[TEST].append(test_saved.accuracy)

        trees_train.append(train_saved.predictions)
        trees_test.append(test_saved.predictions)

        rain_forst_errors[TRAIN].append(evaluateBagging(dataSet, trees_train))
        rain_forst_errors[TEST].append(evaluateBagging(testSet, trees_test))

        if t % 20 == 0:
            print("at", t, "tree")

    print("starting plot")

    x = np.arange(0, T, 1)
    plt.plot(x, rain_forst_errors[TRAIN], label='Training Data for Rain Forest')
    plt.plot(x, rain_forst_errors[TEST], label='Testing Data for Rain Forest')

    plt.plot(x, tree_errors[TRAIN], label='Training Data for Individual')
    plt.plot(x, tree_errors[TEST], label='Testing Data for Individual')

    plt.xlabel("Trees")
    plt.ylabel("Accuracy")

    plt.legend()

    # Add a title
    plt.title("Rain Forest Implementation Accuracy Across Trees with Random Size: " + str(random_set_size))
    plt.show()





if __name__ == '__main__':
    # adaBoost()
    # baggedTrees()
    # randomForest()
    # experimentBaggedTrees()
    experimentRainForest()

