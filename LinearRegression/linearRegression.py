# from utils.helpful import saveDataSet
#use matplot
import math, random
error_threshold = math.pow(10, -6)
maxT = 10000

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

def testWith(w, filename): 
    testData = saveDataSet(filename)
    return totalError(w, testData)

def w_zero(length):
    w = []
    for i in range(length): 
        w.append(0)
    return w

def gradientDescent(w, r, threshold, dataset):
    t = 0
    while totalError(w, dataset) > threshold: 
        gradient = batchGradient(w, dataset)
        w = updateW(w, gradient, r)
        t +=1   
        print(totalError(w, dataset))
    return w

def randomSample(dataset): 
    m = len(dataset)
    i = random.randint(0, m-1)
    return [dataset[i]]

def stochasticDescent(w, r, threshold, dataset):
    t = 0
    while totalError(w, dataset) > threshold: 
        gradSample = randomSample(dataset)
        gradient = batchGradient(w, gradSample)
        w = updateW(w, gradient, r)
        t +=1   
        if t > maxT:
            break 
    return w

def totalError(w, dataset): 
    total = 0
    for x in dataset: 
        error = calcError(x, w)
        total += (error * error)
    return total * (1/2)

#[]
def batchGradient(w, dataset):
    wnext = []
    for j, wj in enumerate(w): 
        error_sum = 0
        for row in dataset: 
            error = calcError(row, w)
            error_sum += error * row[j]
        wnext.append(-error_sum)
    return wnext


def calcError(x, w): 
    yi = x[len(x) - 1]
    error = yi - predict(w, x)
    return error

def predict(w, x): 
    result = 0
    for i, wi in enumerate(w): 
        result += wi * x[i]
    return result

def updateW(w, grad, r): 
    result = []
    for i, wi in enumerate(w): 
        result.append(wi - (r * grad[i]))
    return result


def linearMain(): 
    dataSet = saveDataSet('../datasets/concrete/train.csv')
    wlength = len(dataSet[0]) - 1
    w0 = w_zero(wlength)

    model = gradientDescent(w0, 1, error_threshold, dataSet)
    print(model)
    # print("error", testWith(model, '../datasets/concrete/test.csv'))



if __name__ == '__main__':
    linearMain()