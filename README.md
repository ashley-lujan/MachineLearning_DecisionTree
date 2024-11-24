This is a machine learning library developed by Ashley Lujan for CS5350/6350 in University of Utah. 

# Ensemble Learning
To run the ensemble learning code, run the EnsembleLearning/optimized.py file. This file will run faster than the adaBoost.py file. To avoid unnecessary runs, the code has commented out functions. Uncomment the functions you are interested in running. There are some for adaBoost(), baggedTrees(), randomForest(), and experimentBaggedTrees(). 
If you wish to change the number of trees for a 'bag', change the number T in those functions. The individual decision trees also let you change the max-depth in the constructor as well. 

# Linear Regression
To run the linear regression code, run linearRegression.py. The linearMain() function is where gradient and stochastic are being called. The function descent() allows you to specify what r values you will like to experiment with, along with which specific descent you would like (gradient or stochastic).

# Perceptron
To run the perceptron code, run Perceptron/perceptron.py. If you wish to modify the training and test examples, modify what train_data_filename and test_data_filename have stored in the main function. 