import pandas as pd
import numpy as np 
import pickle 
import csv , math
import sys


all_data = pd.read_csv('/home/morris/Desktop/machine-learning/hw1/csv/train.csv')

all_data_matrix = np.array(all_data)

all_data_matrix = all_data_matrix[:,3:]

(raw_row,raw_col) = all_data_matrix.shape
# let NR equal to 0
for row in range(raw_row):
    for col in range(raw_col):
        if all_data_matrix[row, col] == 'NR':
            all_data_matrix[row, col] = 0
#learning rate


Ground_Truth_list=[]

#the feature number which you want to remove ex: Rm_Feature = [8] ,means i want to remove PM10
Rm_Feature = []
rm_feature = []
for feature in Rm_Feature:
    for data in range(feature, 162, 18):
        rm_feature.append(data)
rm_feature.sort()
rm_feature_number = len(rm_feature)


Rm_Month = []
rm_month = []
for month in Rm_Month:
    rm_month.append(month)
rm_month_number = len(rm_month)


reduce_data = rm_month_number * 471

real_number = 5652 - reduce_data
number_of_weight = 162 - rm_feature_number

#parameter
# weight1 = np.random.uniform(-1,1,(number_of_weight,1))
# weight2 = np.random.uniform(-1,1,(number_of_weight,1))
bias = 0

weight1 = np.zeros((number_of_weight,1))
weight2 = np.zeros((number_of_weight,1))


final_data = np.zeros((real_number , number_of_weight))
Category = np.zeros((20,432))
Category_Data = np.zeros((12 - rm_month_number , 8640))
train = np.zeros((real_number,162))
train_data = all_data_matrix.reshape(12,20,18,24)
i = 0
m = 0

for month in range(12):
    if month in rm_month:
        pass
    else:
        for day in range(20):
            Category[day, :] = train_data[m, day,:,:].T.reshape(1, 432)
            if day == 0:
                for hours in range(9, 24):
                    Ground_Truth_list.append(train_data[m, day, 9, hours])
            else:
                for hours in range(24):
                    Ground_Truth_list.append(train_data[m, day, 9, hours])

        Category_Data[m, :] = Category[:, :].reshape(1, 8640)

        for time in range(0, 8478, 18):
            n = 0
            for feature in range(time, time + 162):
                train[i, n] = Category_Data[m, feature]
                n += 1
            i += 1
        m +=1   

#the number of weight
num = 0
for x in range(162):
    if x in rm_feature:
        pass
    else:
        final_data[:,num] = train[:,x]
        num += 1

#normalization
std = np.std(final_data,axis=0)
means = np.mean(final_data,axis=0).round(decimals=5)

Ground_Truth_list_combine_final_data = np.zeros((real_number,number_of_weight + 1))
epilson = 1e-5
for row in range(real_number):
    for col in range(number_of_weight):
        if std[col] ==0:
                final_data[row,col] = 0
        else:
            final_data[row,col] = (final_data[row,col] - means[col]) / (std[col] + epilson)

for row in range(real_number):
    Ground_Truth_list_combine_final_data[row,0] = Ground_Truth_list[row]
    Ground_Truth_list_combine_final_data[row,1:] = final_data[row,:]

Ground_Truth_matrix = np.zeros((real_number,1))

np.random.shuffle(Ground_Truth_list_combine_final_data)
Ground_Truth_matrix = Ground_Truth_list_combine_final_data[:,0].reshape(real_number,1)
final_data = Ground_Truth_list_combine_final_data[:,1:]


lr = 1e-3
iteration = 30001
train_number = int(real_number * 0.9)
validation_number = real_number - train_number

for i in range(iteration):
    delta_weight1 = np.zeros((number_of_weight,1))
    delta_weight2 = np.zeros((number_of_weight,1))        
    delta_bias = 0
    time = 0
    train_loss = 0
    if i % 5000 == 0:
        lr = lr / 2
    for data in range(0, train_number):

        double_x = final_data[data, :].T ** 2
        predict_value = bias + np.dot(weight1.T, final_data[data, :].T) + np.dot(weight2.T, double_x)
        loss = Ground_Truth_matrix[data, 0] - predict_value[0]
        delta_weight1 = delta_weight1 - 2 * (loss * final_data[data, :].reshape(number_of_weight, 1))
        delta_weight2 = delta_weight2 - 2 * (loss * double_x.reshape(number_of_weight, 1))
        delta_bias = delta_bias - 2 * loss
        time += 1
        train_loss += (loss ** 2)
        if time % int(train_number / 5) == 0:
            weight1 = weight1 - (lr * (delta_weight1 / (int(train_number / 5))))
            weight2 = weight2 - (lr * (delta_weight2 / (int(train_number / 5))))
            bias = bias - (lr * (delta_bias / (int(train_number / 5))))
            delta_weight1 = np.zeros((number_of_weight, 1))
            delta_weight2 = np.zeros((number_of_weight, 1))
            delta_bias = 0
            time = 0
        elif data == train_number - 1:
            weight1 = weight1 - (lr * delta_weight1 / (train_number - (int(train_number / 5))* 5))
            weight2 = weight2 - (lr * delta_weight2 / (train_number - int(train_number / 5) * 5))
            bias = bias - (lr * delta_bias / (train_number - (int(train_number / 5) * 5)))
            delta_weight1 = np.zeros((number_of_weight, 1))
            delta_weight2 = np.zeros((number_of_weight, 1))
            delta_bias = 0
            time = 0
    b = 0.0
    for val in range(real_number - validation_number, real_number):
        double_x = final_data[val, :] ** 2
        dot = np.dot(weight1.T, final_data[val, :].T) + np.dot(weight2.T, double_x.T)
        loss = Ground_Truth_matrix[val,0] - ( dot[0] + bias) 

        a = loss ** 2
        b += a
    RMSE = math.sqrt(b / validation_number)
    
    sys.stdout.write('\r{} {:7.5f} {:7.5f}'.format(i, math.sqrt(train_loss / train_number), RMSE) )
    sys.stdout.flush()


np.savetxt('/home/morris/Desktop/machine-learning/hw1/txt/num_onlypm.txt', np.array(number_of_weight).reshape(1,))
np.savetxt('/home/morris/Desktop/machine-learning/hw1/txt/weight1_onlypm.txt', weight1)
np.savetxt('/home/morris/Desktop/machine-learning/hw1/txt/weight2_onlypm.txt', weight2)
np.savetxt('/home/morris/Desktop/machine-learning/hw1/txt/bias_onlypm.txt', np.array(bias).reshape(1,))
np.savetxt('/home/morris/Desktop/machine-learning/hw1/txt/mean_onlypm.txt', means)
np.savetxt('/home/morris/Desktop/machine-learning/hw1/txt/std_onlypm.txt', std)