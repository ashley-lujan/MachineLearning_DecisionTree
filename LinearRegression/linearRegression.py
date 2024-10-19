# from utils.helpful import saveDataSet
#use matplot
import math, random
import numpy as np
import matplotlib.pyplot as plt
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
    prevError = math.inf
    currentError = totalError(w, dataset)
    cost_values = [currentError]
    while abs(currentError - prevError) > threshold:
        gradient = batchGradient(w, dataset)
        w = updateW(w, gradient, r)
        t +=1
        prevError = currentError
        currentError = totalError(w, dataset)
        cost_values.append(currentError)
    return cost_values

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


def descent(dataSet, r_lst, title, descentFunc):
    wlength = len(dataSet[0]) - 1

    w0 = w_zero(wlength)

    for r in r_lst:
        cost_values = descentFunc(w0, r, error_threshold, dataSet)
        x = np.arange(0, len(cost_values), 1)
        plt.plot(x, cost_values)
        plt.xlabel('Iteration')
        plt.ylabel('Total Error')

        # Add a title
        plt.title(title + ' Total Error at Each Iteration for R = ' + str(r))
        plt.show()


def linearMain():
    dataSet = saveDataSet('../datasets/concrete/train.csv')
    descent(dataSet, [1, 0.5, 0.25, 0.125, 0.01], 'Batch Gradient', gradientDescent)
    descent(dataSet, [0.125, 0.01, 0.001, 0.0001, 0.00001, 0.000001], 'Stochastic Gradient', stochasticDescent)







if __name__ == '__main__':
    linearMain()
    # linearMain([1])