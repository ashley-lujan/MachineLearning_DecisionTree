from weightedSet import WeightedX
from weightedDecAlgo import DecisionTree

#returns a list of weightedX's
def saveDataSet(filename):
    result = []
    with open(filename, 'r') as f:
        for line in f:
            terms = line.strip().split(',')
            row = []
            for el in terms:
                row.append(el)
            result.append(WeightedX(1, row))
    return result

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
    dataSet = saveDataSet('../datasets/bank/train.csv')
    d3 = DecisionTree(dataSet=dataSet, attributesWithValues=getBankAttributes(), hasNumerics=True, max_depth=2, replaceMissing=True)
    print("uh")
    d3.printTree(d3.root, "")

if __name__ == '__main__':
    adaBoost()