# from utils.helpful import saveDataSet
import math
error_threshold = math.pow(10, -6)


def saveDataSet(filename):
    result = []
    with open(filename, 'r') as f:
        for line in f:
            terms = line.strip().split(',')
            row = [1]
            for el in terms:
                row.append(float(el))
            result.append(row)
    return result

def w_zero(length):
    w = []
    for i in range(length): 
        w.append(0)
    return w

#should return w
def gradientDescent(w, r, threshold, dataset):
    while totalError(w, dataset) > threshold: 
        gradient = batchGradient(w, dataSet)
        break
    return w

def totalError(w, dataset): 
    return 0

def batchGradient(w, dataset):
    wnext = []
    for j, wj in enumerate(w): 
        error_sum = 0
        for row in dataset: 
            yi = row[len(row) - 1]
            error = yi - predict(w, row)
            error_sum += error * row[j]
        wnext.append(-error_sum)
    return wnext

def predict(w, x): 
    result = 0
    for wi in w:
        for xi in x[0::-1]: 
            result += wi * xi
    return result

def gradient(w, row):
    return 0

def linearMain(): 
    dataSet = saveDataSet('concrete/train.csv')
    wlength = len(dataSet[0]) - 1
    w0 = w_zero(wlength)

    model = batchGradient(w0, dataSet)
    print(model)





if __name__ == '__main__':
    # print([0, 1, 2, 3][:-1])
    linearMain()