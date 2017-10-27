import pandas as pd
import numpy as np
import function
import sys
data = function.get_data(sys.argv[3])
row ,col = data.shape
training_data = np.zeros((row, 6))

label =function.get_data(sys.argv[4])

test_data = function.get_data(sys.argv[5])
number_of_data, number_of_weight = test_data.shape
testing_data = np.zeros((number_of_data, 6))

n = 0
for i in range(6):
    training_data[:, n] = data[:, i]
    testing_data[:, n] = test_data[:, i]
    n += 1
training_data = function.min_max_normalize(training_data)
mini = np.loadtxt('min.txt')
maxi = np.loadtxt('max.txt')
need_normal = [0, 2, 3, 4]
n = 0
for i in need_normal:
    for j in range(number_of_data):
        testing_data[j, i] = (testing_data[j, i] - mini[n]) / (maxi[n] - mini[n])
    n += 1

class0,class1 = function.classification(label,training_data)
number_class0,weight = class0.shape
number_class1,weight = class1.shape
Pro_C0 = float(number_class0) /float(number_class0 + number_class1)
Pro_C1 = float(number_class1) /float(number_class0 + number_class1)
Gaussuan_0 = function.Generative_model()
Gaussuan_0.Maximum_Likelihood(class0)

Gaussuan_1 = function.Generative_model()
Gaussuan_1.Maximum_Likelihood(class1)

Gaussuan_0.Gaussian_Distrobution(testing_data[0,:])
Gaussuan_1.Gaussian_Distrobution(testing_data[0,:])

test_id = []
predict_value = []

for item in range(number_of_data): 
    test_id.append(str(item + 1))
    probability_0 = (Gaussuan_0.Gaussian_Distrobution(testing_data[item,:]))
    probability_1 = (Gaussuan_1.Gaussian_Distrobution(testing_data[item,:]))   
    probability = (probability_1 * Pro_C1)/((probability_1 * Pro_C1) + (probability_0 * Pro_C0))

    if probability >= 0.5:
        predict = 1
    else:
        predict = 0
    predict_value.append(predict)
df = pd.DataFrame({'id': test_id, 'label': predict_value})
df.to_csv(sys.argv[6], index=False)