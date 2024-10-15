# import pandas as pd
from decAlgo import Node, DecisionTree
import math

# def processAttributesAndValues(filename):
#     df = pd.read_csv(filename + '/train.csv', header=None)
#     attributes = getCarAttributes(filename)
#     for i in range(0, len(attributes)):
#         values = df[i].value_counts()
#         print("for attribute", attributes[i], " i have these values: ", values)
#     return df

def getCarAttributes(filename) :
    car_attributes = {
        "buying": ["vhigh", "high", "med", "low"],
        "maint": ["vhigh", "high", "med", "low"],
        "doors": ["2", "3", "4", "5more"],
        "persons": ["2", "4", "more"],
        "lug_boot": ["small", "med", "big"],
        "safety": ["low", "med", "high"]
    }
    return car_attributes

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

def evaluateModel(d3, packet, csv):
    testSet = saveDataSet('datasets/' + packet + '/' + csv + '.csv')
    correct = 0
    for test in testSet:
        prediction = d3.predict(d3.root, test)
        if prediction == test[len(test) - 1]:
            correct += 1
    return correct/len(testSet)


def question2_1():
    #Question 2.1
    dataSet = saveDataSet('datasets/car/train.csv')
    l1 = getCarAttributes("s")
    calcs = ["Ent", "ME", "G"]
    print("\t \t \t", end="")
    for i in range(1, 7):
        print(i, end="\t")
    print("average", "")
    for measure in calcs:
        print("for ", measure, ": ")
        print("train", end="\t")
        average = 0
        for i in range(1, 7):
            dataSet = saveDataSet('datasets/car/train.csv')
            d3 = DecisionTree(dataSet, l1, False, i, measure)
            accuracy = evaluateModel(d3, 'car','train')
            average += accuracy
            print("", round(accuracy, 3),  end=" & ")
        print(round(average / 6, 3))
        print("test", end="\t")
        average = 0
        for i in range(1, 7):
            dataSet = saveDataSet('datasets/car/train.csv')
            d3 = DecisionTree(dataSet, l1, False, i, measure)
            accuracy = evaluateModel(d3, 'car', 'test')
            average += accuracy
            print("", round(accuracy, 3), end= " & ")
        print(round(average / 6, 3))
        print("")

def question2_2_1():
    calcs = ["Ent", "ME", "G"]
    print("dataset\t Measurement \t average")
    max_depth = 17
    for measure in calcs:
        # print("for ", measure, ": ")
        # print("test", end="\t")
        average = 0
        for i in range(1, max_depth):
            dataSet = saveDataSet('datasets/bank/train.csv')
            attrs = getBankAttributes()
            d3 = DecisionTree(dataSet, attrs, True, i, measure)
            accuracy = evaluateModel(d3, 'bank', 'train')
            average += accuracy
            print("", round(accuracy, 3), end="\t")
        print("\ntrain & ", measure, " & ", round(average / (max_depth - 1), 3), " \\\\ \\hline")
        # print("train", end="\t")
        average = 0
        for i in range(1, max_depth):
            dataSetTrain = saveDataSet('datasets/bank/train.csv')
            attrs = getBankAttributes()
            d3 = DecisionTree(dataSetTrain, attrs, True, i, measure)
            accuracy = evaluateModel(d3, 'bank', 'test')
            average += accuracy
            print("", round(accuracy, 3), end="\t")
        print("\ntest & ", measure, " & ", round(average / (max_depth - 1), 3), " \\\\ \\hline")
        print("")

def question2_2_2():
    calcs = ["Ent", "ME", "G"]
    print("\t \t \t", end="")
    max_depth = 17
    print("dataset\t Measurement \t average")
    for measure in calcs:
        average = 0
        for i in range(1, max_depth):
            dataSet = saveDataSet('datasets/bank/train.csv')
            attrs = getBankAttributes()
            d3 = DecisionTree(dataSet, attrs, True, i, measure, True)
            accuracy = evaluateModel(d3, 'bank', 'train')
            average += accuracy
            print("", round(accuracy, 3), end="\t")
        print("\n train & ", measure, " & ", round(average / (max_depth - 1), 3), " \\\\ \\hline")
        # print("train", end="\t")
        average = 0
        for i in range(1, max_depth):
            dataSetTrain = saveDataSet('datasets/bank/train.csv')
            attrs = getBankAttributes()
            d3 = DecisionTree(dataSetTrain, attrs, True, i, measure, True)
            accuracy = evaluateModel(d3, 'bank', 'test')
            average += accuracy
            print("", round(accuracy, 3), end="\t")
        print("\n test & ", measure, " & ", round(average / (max_depth - 1), 3), " \\\\ \\hline")
        print("")

if __name__ == '__main__':
    print("question 2.1 with car")
    question2_1()
    print("question 2.2.a with bank")
    question2_2_1()
    print("question 2.2.b with bank")
    question2_2_2()


