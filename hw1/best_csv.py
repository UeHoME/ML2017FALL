import pandas as pd 
import numpy as np
import math
import sys

weight1 = np.loadtxt('weight1_best.txt')
weight2 = np.loadtxt('weight2_best.txt')
bias = np.loadtxt('bias_best.txt')
means = np.loadtxt('mean_best.txt')
std = np.loadtxt('std_best.txt')
number_of_weight = np.loadtxt('num_best.txt')


data = pd.read_csv(sys.argv[1],header = None)
data = np.array(data)

data_matrix = np.zeros((240, 162))
final_data = np.zeros((240, int(number_of_weight)))
data[data == 'NR'] = 0.0

matrix = data[:, 2:].reshape(240, 18, 9).astype(float)

Rm_Feature = [0, 1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15, 16, 17]
rm = []
for feature in Rm_Feature:
    for data in range(feature, 162, 18):
        rm.append(data)
rm.sort()

for i in range(240):
    data_matrix[i, :] = matrix[i, :, :].T.reshape(1, 162)

for row in range(240):
    n = 0
    for col in range(162):
        if col in rm:
            pass
        else:
            final_data[row, n] = data_matrix[row, col]
            n += 1

epilson = 1e-5
for row in range(240):
    for col in range(int(number_of_weight)):
            final_data[row, col]  = (final_data[row, col] - means[col]) / (std[col] + epilson)

predict_value = []
test_id = []
for item in range(240):
    test_id.append('id_'+str(item))
    test_matrix = final_data[item, :] .T.astype(float).reshape(1, int(number_of_weight))
    double = test_matrix[0, :] ** 2
    predict = np.dot(double, weight2) + np.dot(test_matrix[0, :], weight1) + bias
    predict_value.append(predict)

df = pd.DataFrame({'id': test_id, 'value': predict_value})
df.to_csv(sys.argv[2], index=False)