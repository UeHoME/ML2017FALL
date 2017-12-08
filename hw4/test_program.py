import pickle
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.preprocessing import sequence
import sys

max_word_idx = 100000
max_sequence_len = 50
embedding_vector_len = 50
threshold = 0.5

with open('tokenizer.pickle', 'rb') as handle:
    # x_test = pickle.load(handle)
    tokenizer =  pickle.load(handle)

x_test = [line.rstrip('\n') for line in open(sys.argv[1], 'r')]
x_test = [line.split(',', 1)[1] for line in x_test]
del x_test[0]

model = load_model('model_0.h5')
# t = ["today is a good day, but it is hot", "today is hot, but it is a good day"]
x_test = tokenizer.texts_to_sequences(x_test)
x_test = sequence.pad_sequences(x_test, maxlen = max_sequence_len)
x_test = np.asarray(x_test)
# t = sequence.pad_sequences(t, maxlen = max_sequence_len)
# t = np.asarray(t)
predict = model.predict(x_test).reshape(-1,1)
# print(predict)
predict = (predict > threshold).astype(int)
value = []
id_col = []
for i in range(predict.shape[0]):
	a = predict[i]
	value.append(a[0])
	id_col.append(i)

output = pd.DataFrame({'id': id_col , 'label':value})
output.to_csv(sys.argv[2], index = False)