import math
import statistics

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


class WeightedDecisionTree:
    #don't change value of self.attributes --> maintains index of what columns are
    less = "0"
    greaterE = "1"


    def __init__(self, dataSet : list, attributesWithValues : dict, dataMedians : dict, max_depth, replaceMissing = False,):
        self.root = Node(1)
        self.attributes = attributesWithValues
        self.hasNumerics = True
        self.medians = dataMedians
        # if hasNumerics:
        #     self.processDataSet(dataSet, attributesWithValues)
        if replaceMissing:
            self.replaceUnknown(dataSet, attributesWithValues)
        self.formDecision(self.root, dataSet, (list)(attributesWithValues.keys()), max_depth)
        
        self.at = self.computeAt(dataSet)

    def reweigh_dataSet(self, dataset): 
        normalization = 0
        for x in dataset: 
            dnext = x.weight * math.pow(math.e, (-1 * self.at * self.correctness(x)))
            normalization += dnext
            x.weight = dnext

        for x in dataset: 
            normalized_weight = x.weight/normalization
            x.weight = normalized_weight

    def computeAt(self, dataset): 
        et = self.computeEt(dataset)
        return (1/2) * math.log((1 - et)/et)

    def computeEt(self, dataSet): 
        sum = 0
        for x in dataSet: 
            sum += (x.weight * self.correctness(x))
        return (1/2) - ((1/2) * sum)
    
    #returns yi * ht(xi)
    def correctness(self, x):
        x_data = x.data
        return self.translate(x_data[len(x_data) -1]) * self.translate(self.predict(x.data))
    
    def translate(self, value): 
        if value == "yes": 
            return 1
        return -1
    

    def replaceUnknown(self, dataSet, attrs):
        for i, attr in enumerate(attrs):
            if "unknown" in attrs[attr]:
                mostCommon = self.mostCommon(dataSet, i, attrs[attr])
                self.replaceCols(dataSet, i,"unknown", mostCommon)

    def mostCommon(self, dataSet, i, values):
        track = {}
        for v in values:
            track[v] = 0

        for row in dataSet:
            value = row.data[i]
            track[value] = track[value] + 1

        return max(track, key=track.get)

    def replaceCols(self, dataSet, i, wordToReplace, replacer):
        for row in dataSet:
            if row.data[i] == wordToReplace:
                row.data[i] = replacer



    def processDataSet(self, dataSet, attrs):
        for i, attr in enumerate(attrs):
            if attrs[attr] == "numeric":
                med = self.findMedian(dataSet, i)
                self.medians[i] = med
                self.replaceWithBinary(dataSet, i, med)
                attrs[attr] = [self.less, self.greaterE]

    def findMedian(self, dataSet, i):
        l = []
        for row in dataSet:
            l.append(int(row.data[i]))
        return statistics.median(l)

    def replaceWithBinary(self, dataSet, i, med):
        for row in dataSet:
            val = int(row.data[i])
            if val >= med:
                row.data[i] = self.greaterE
            else:
                row.data[i] = self.less



    def measurer(self, l):
        return self.calcEntropyS(l)
 


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
                    self.formDecision(branch, sv, self.removeAttribute(current_attributes, A), max_depth)

    def removeAttribute(self, atts, A):
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

        s_size = self.sumOfS(s)

        for A in current_attribute:
            expected_reduction = 0
            for value in self.attributes[A]:
                sv = self.subsetWithValue(s, A, value)
                sum_of_sv = self.sumOfS(sv)
                expected_reduction += (sum_of_sv/s_size) * self.measurer(sv)
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
            p = tracker[label]  / sampleSize
            sum = p * math.log(p, 2)

        return -sum
    
    def sumOfS(self, s : list): 
        sum = 0
        for x in s: 
            sum += x.weight
        return sum


    def subsetWithValue(self, s, A, value):
        index = (list)(self.attributes.keys()).index(A)
        subset = []
        for row in s:
            if row.data[index] == value:
                subset.append(row)
        return subset

    def sameLabel(s, l):
        lastIndex = len(l[0].data) - 1
        label = l[0].data[lastIndex]
        for item in l:
            if item.data[lastIndex] != label:
                return None
        return label

    def trackerOfLabels(s, l):
        tracker = {}
        lastIndex = len(l[0].data) - 1
        for item in l:
            label = item.data[lastIndex]
            tracker[label] = tracker.get(label, 0) + item.weight
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

    

    def getValue(self, l, index):
        if self.hasNumerics and index in self.medians:
            if int(l[index]) < self.medians[index]:
                return self.less
            else:
                return self.greaterE
        else:
            return l[index]


    def predict(self, l):
        return self.predict_tree(self.root, l)

    
    def predict_tree(self, node, l): 
        if node.isLeafNode:
            return node.label
        else:
            index = (list)(self.attributes.keys()).index(node.attribute)
            rowValue = self.getValue(l, index)
            nextBranch = node.branches[rowValue]
            return self.predict_tree(nextBranch, l)




