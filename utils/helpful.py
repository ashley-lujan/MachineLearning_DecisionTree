def saveDataSet(filename):
    result = []
    # with open(filename + '/train.csv', 'r') as f:
    with open(filename, 'r') as f:
        for line in f:
            terms = line.strip().split(',')
            result.append(terms)
    return result
