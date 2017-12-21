import csv
import sys
import numpy as np
import pandas as pd
from keras.models import load_model
from keras import backend as K

def load_data(fileName):
    dataFrame = pd.read_csv(fileName)
    user_id = dataFrame['UserID']
    user_id = user_id.apply(pd.to_numeric).as_matrix()-1
    movie_id = dataFrame['MovieID']
    movie_id = movie_id.apply(pd.to_numeric).as_matrix()-1
    return user_id, movie_id

# load training data
user_id, movie_id = load_data(sys.argv[1])
print("Finish reading data")

model = load_model("model_MF_64.h5")

result = model.predict([user_id, movie_id], verbose = 1)
# ================== normalize ================= #

# result = result.argmax(axis=-1)
# for i in range(len(result)):
# 	result[i] = (result[i] * 4) + 1

# ================== id ======================= #
id_data = []
op = []
print("\n")
print("Finish predict")
for i in range(len(result)):
	id_data.append(i + 1)
	a = result[i]
	op.append(a[0])

output = pd.DataFrame({'TestDataID': id_data,'Rating':op})
output.to_csv(sys.argv[2], index = False)