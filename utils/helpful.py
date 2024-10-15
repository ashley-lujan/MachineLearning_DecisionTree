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
