import concurrent.futures
import math
from decAlgo import DecisionTree
from optimized import getBankAttributes, saveDataSet, randomizeSet, translate, calcFinal, computeSampleVariance

experiment_runs = (104 // 8 )
training_size = 1000
trees_to_learn = 500

bagged_trees = []
single_trees = []

# Load datasets
dataSet = saveDataSet('../datasets/bank/train.csv')
testSet = saveDataSet('../datasets/bank/test.csv')



# Function to run a single experiment run for training
def train_run(run):
    # Initialize lists for this run
    baggedSet = randomizeSet(dataSet, training_size)

    # Train single decision tree
    single_tree = DecisionTree(dataSet=baggedSet, attributesWithValues=getBankAttributes(), hasNumerics=True,
                               max_depth=math.inf, replaceMissing=True)

    # Train bagged trees
    trees = []
    for tree_to_learn in range(trees_to_learn):
        d3 = DecisionTree(dataSet=baggedSet, attributesWithValues=getBankAttributes(), hasNumerics=True,
                          max_depth=math.inf, replaceMissing=True)
        trees.append(d3)

    return single_tree, trees


if __name__ == "__main__":
    single_trees = []
    predictors = []

    # Parallel training of the trees
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(train_run, run) for run in range(experiment_runs)]

        for idx, future in enumerate(concurrent.futures.as_completed(futures)):
            single_tree, trees = future.result()
            single_trees.append(single_tree)
            predictors.append(trees)

            print(f"Completed training for run {idx}")
        print("joined")

    # Sequential evaluation after training
    # Single tree evaluation
    sngle_tree_average_bias = 0
    sngle_tree_average_var = 0
    for test in testSet:
        average = 0
        test_predictions = []
        for sngle_tree in single_trees:
            predic = translate(sngle_tree.predict(test))
            test_predictions.append(predic)
            average += predic
        average = average / len(single_trees)
        expected = translate(test[-1])
        bias = math.pow(average - expected, 2)
        sample_var = computeSampleVariance(expected, test_predictions)
        sngle_tree_average_bias += bias
        sngle_tree_average_var += sample_var
    sngle_tree_average_var /= len(single_trees)
    sngle_tree_average_bias /= len(single_trees)
    sngle_squared_error = sngle_tree_average_bias + sngle_tree_average_var
    print("Single Trees")
    print(sngle_tree_average_bias, sngle_tree_average_var, sngle_squared_error)

    # Bagged trees evaluation
    bagged_tree_average_bias = 0
    bagged_tree_average_var = 0
    for test in testSet:
        average = 0
        test_predictions = []
        for bag in predictors:
            predic = translate(calcFinal(bag, test, 0))
            test_predictions.append(predic)
            average += predic
        average = average / len(predictors)
        expected = translate(test[-1])
        bias = math.pow(average - expected, 2)
        sample_var = computeSampleVariance(expected, test_predictions)
        bagged_tree_average_bias += bias
        bagged_tree_average_var += sample_var
    bagged_tree_average_var /= len(predictors)
    bagged_tree_average_bias /= len(predictors)
    bagged_squared_error = bagged_tree_average_bias + bagged_tree_average_var
    print("Bagged Trees")
    print(bagged_tree_average_bias, bagged_tree_average_var, bagged_squared_error)
