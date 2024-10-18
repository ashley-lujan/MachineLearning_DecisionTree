from decAlgo import DecisionTree
from random import randint

class RandomForestTree(DecisionTree):
    def __init__(self, feature_subset_size: int, dataSet : list, attributesWithValues : dict, hasNumerics : bool, max_depth,  replaceMissing = False): 
        self.feature_subset_size = feature_subset_size
        super().__init__(dataSet, attributesWithValues, hasNumerics, max_depth, replaceMissing)

    def formDecision(self, node, s : list, current_attributes : list, max_depth):
        if len(current_attributes) == 0 or node.level >= max_depth:
            node.setLabel(self.mostCommonLabel(s)) #is this a case he didn't talk about?
            return
        entropy_attributes = self.getSubset(current_attributes)
        A = self.bestA(s, entropy_attributes)
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
    
    def getSubset(self, lst): 
        m = len(lst)
        subset = []
        for i in range(self.feature_subset_size): 
            index = randint(0, m-1)
            random_element = lst[index]
            subset.append(random_element)
        return subset
