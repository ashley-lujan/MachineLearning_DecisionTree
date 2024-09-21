import pandas as pd
import math

class Node:
    def __init__(self, level ):
        self.isLeafNode = False
        self.attribute = None
        self.label = None
        self.level = level
        # value of attribute : Node
        self.branches = {}

    def addBranch(self, value):
        branch = Node(self.level + 1)
        self.branches[value] = branch
        return branch

    def setLabel(self, label):
        self.isLeafNode = True
        self.label = label


class DecisionTree:
    #don't change value of self.attributes --> maintains index of what columns are

    def __init__(self, dataSet : list, attributesWithValues : dict, max_depth, measurement):
        self.root = Node(1)
        self.attributes = attributesWithValues
        self.measurement = measurement
        self.formDecision(self.root, dataSet, (list)(attributesWithValues.keys()), max_depth)

    def __init__(self, dataSet : list, attributesWithValues : dict, updateLabel : int,  max_depth, measurement):
        self.root = Node(1)
        self.attributes = attributesWithValues
        self.measurement = measurement
        #preprocess dataset at updateLabel
        self.formDecision(self.root, dataSet, (list)(attributesWithValues.keys()), max_depth)


    def measurer(self, l):
        if self.measurement == "ME":
            return self.calcME(l)
        if self.measurement == "Ent":
            return self.calcEntropyS(l)
        if self.measurement == "G":
            return self.calcGini(l)
        else:
            Exception("invalid measurement")


    def formDecision(self, node, s : list, current_attributes : list, max_depth):
        if len(current_attributes) == 0 or node.level >= max_depth:
            node.setLabel(self.mostCommonLabel(s)) #is this a case he didn't talk about?
            return
        A = self.bestA(s, current_attributes)
        node.attribute = A
        for value in self.attributes[A]:
            branch = node.addBranch(value)
            sv = self.subsetWithValue(s, A, value)

            if len(sv) == 0:
                commonLabel = self.mostCommonLabel(s)
                branch.setLabel(commonLabel)
            else:
                uniformLabel = self.sameLabel(sv)
                if uniformLabel:
                    branch.setLabel(uniformLabel)
                else:
                    self.formDecision(branch, sv, self.removeItem(current_attributes, A), max_depth)

    def removeItem(self, atts, A):
        result = []
        for item in atts:
            if item != A:
                result.append(item)
        # print("afer removing item", result)
        return result


    def bestA(self, s : list, current_attribute: list):
        # {attribute: information gain}
        information_gain = {}
        entropyS = self.measurer(s)

        for A in current_attribute:
            expected_reduction = 0
            for value in self.attributes[A]:
                sv = self.subsetWithValue(s, A, value)
                expected_reduction += (len(sv)/len(s)) * self.measurer(sv)
            information_gain[A] = entropyS - expected_reduction

        # print(information_gain)
        return max(information_gain, key=information_gain.get)
        # if len(current_attribute) > 0:
        # return current_attribute[0] #todo: implement this later

    def calcEntropyS(self, s : list):
        if len(s) == 0:
            return 0

        tracker = self.trackerOfLabels(s)
        sampleSize = len(s)

        sum = 0
        for label in tracker:
            p = tracker[label] / sampleSize
            sum = p * math.log(p, 2)

        return -sum

    def calcME(self, s:list):
        if len(s) == 0:
            return 0

        tracker = self.trackerOfLabels(s)
        sampleSize = len(s)

        for label in tracker:
            p = tracker[label] / sampleSize
            tracker[label] = p

        majorityLabel = max(tracker, key=tracker.get)
        majority = tracker[majorityLabel]

        return 1 - majority

    def calcGini(self, s:list):
        if len(s) == 0:
            return 0

        tracker = self.trackerOfLabels(s)
        sampleSize = len(s)

        giniIndex = 0
        for label in tracker:
            p = tracker[label] / sampleSize
            giniIndex += p * p
        return 1 - giniIndex



    def subsetWithValue(self, s, A, value):
        index = (list)(self.attributes.keys()).index(A)
        subset = []
        for row in s:
            if row[index] == value:
                subset.append(row)
        return subset

    def sameLabel(s, l):
        lastIndex = len(l[0]) - 1
        label = l[0][lastIndex]
        for item in l:
            if item[lastIndex] != label:
                return None
        return label

    def trackerOfLabels(s, l):
        tracker = {}
        lastIndex = len(l[0]) - 1
        for item in l:
            label = item[lastIndex]
            tracker[label] = tracker.get(label, 0) + 1
        return tracker

    def mostCommonLabel(s, l):
        tracker = s.trackerOfLabels(l)

        return max(tracker, key=tracker.get)

    def printTree(self, node, str):
        if node.isLeafNode:
            print(str, node.label)
            return

        print(str, node.attribute, "{")

        for b in node.branches:
            self.printTree(node.branches[b], b)

        print("}")

    def predict(self, node, l):
        if node.isLeafNode:
            return node.label
        else:
            index = (list)(self.attributes.keys()).index(node.attribute)
            rowValue = l[index]
            nextBranch = node.branches[rowValue]
            return self.predict(nextBranch, l)




