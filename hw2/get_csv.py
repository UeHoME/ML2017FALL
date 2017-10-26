import numpy as np
import pandas as pd
import function
import math
import sys
test_data = function.get_data(sys.argv[5])

w1 = np.loadtxt('w1.txt')
w2 = np.loadtxt('w2.txt')
b = np.loadtxt('b1.txt')
mini = np.loadtxt('min.txt')
maxi = np.loadtxt('max.txt')

number_of_data, number_of_weight = test_data.shape
testing_data = np.zeros((number_of_data, number_of_weight - 1))
n = 0
for i in range(number_of_weight):
    if i !=1:
        testing_data[:, n] = test_data[:, i]
        n += 1

need_normal = [0, 2, 3, 4]
n = 0
for i in need_normal:
    for j in range(number_of_data):
        testing_data[j, i] = (testing_data[j, i] - mini[n]) / (maxi[n] - mini[n])
    n += 1

test_id = []
predict_value = []
for item in range(number_of_data):
    test_id.append(str(item + 1))
    test_matrix = testing_data[item, :] .T.astype(float).reshape(1, int(number_of_weight - 1))
    double = test_matrix ** 2
    z = np.dot(test_matrix[0, :], w1) + b + np.dot(double,w2)

    predict = 1 / (1 + math.exp(-z))
    if predict >= 0.5:
        predict = 1
    else:
        predict = 0
    predict_value.append(predict)


df = pd.DataFrame({'id': test_id, 'label': predict_value})
df.to_csv(sys.argv[6], index=False)