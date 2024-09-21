import pandas as pd
from decAlgo import Node, DecisionTree
import math

def processAttributesAndValues(filename):
    df = pd.read_csv(filename + '/train.csv', header=None)
    attributes = getAttributes(filename)
    for i in range(0, len(attributes)):
        values = df[i].value_counts()
        print("for attribute", attributes[i], " i have these values: ", values)
    return df

def getAttributes(filename) :
    # with open(filename + '/data-desc.txt', 'r') as f:
    #     for line in f:
    #         print(line.strip())
    # if (filename == )
    car_attributes = {
        "buying": ["vhigh", "high", "med", "low"],
        "maint": ["vhigh", "high", "med", "low"],
        "doors": ["2", "3", "4", "5more"],
        "persons": ["2", "4", "more"],
        "lug_boot": ["small", "med", "big"],
        "safety": ["low", "med", "high"]
    }
    return car_attributes
    # return [
    #     "age",
    #     "job",
    #     "marital",
    #     "education",
    #     "default",
    #     "balance",
    #     "housing",
    #     "loan",
    #     "contact",
    #     "day",
    #     "month",
    #     "duration",
    #     "campaign",
    #     "pdays",
    #     "previous",
    #     "poutcome"
    # ]
def saveDataSet(filename):
    result = []
    inti = 0
    # with open(filename + '/train.csv', 'r') as f:
    with open(filename, 'r') as f:
        for line in f:
            terms = line.strip().split(',')
            inti += 1
            result.append(terms)
    return result

def evaluateModel(d3, csv):
    testSet = saveDataSet('datasets/car/' + csv + '.csv')
    correct = 0
    for test in testSet:
        prediction = d3.predict(d3.root, test)
        if prediction == test[len(test) - 1]:
            correct += 1
    return correct/len(testSet)


if __name__ == '__main__':
    #Question 2.1
    dataSet = saveDataSet('datasets/car/train.csv')
    l1 = getAttributes("s")
    calcs = ["Ent", "ME", "G"]
    print("\t \t \t", end="")
    for i in range(1, 6):
        print(i, end="\t")
    print("average", "")
    for measure in calcs:
        print("for ", measure, ": ")
        print("test", end="\t")
        average = 0
        for i in range(1, 7):
            dataSet = saveDataSet('datasets/car/test.csv')
            d3 = DecisionTree(dataSet, l1, i, measure)
            accuracy = evaluateModel(d3, 'train')
            average += accuracy
            print("", round(accuracy, 3), end="\t")
        print(round(average/6, 3))
        print("train", end="\t")
        average = 0
        for i in range(1, 6):
            dataSet = saveDataSet('datasets/car/train.csv')
            d3 = DecisionTree(dataSet, l1, i, measure)
            accuracy = evaluateModel(d3, 'train')
            average += accuracy
            print("", round(accuracy, 3), end="\t")
        print(round(average / 6, 3))
        print("")


    # d3.printTree(d3.root, "root")


    # print("I got this number correct", correct/len(testSet))

    # print(d3.predict(d3.root, ["vhigh","vhigh","5more","4","big","med","acc"]))
    # print("hello world!")
    # result = processAttributesAndValues('datasets/bank')

