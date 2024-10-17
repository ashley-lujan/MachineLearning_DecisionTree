from weightedSet import WeightedX

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

def adaBoost():
    data = saveDataSet('../datasets/bank/train.csv')
    print(data[0].weight, data[0].data)

if __name__ == '__main__':
    adaBoost()