import numpy as np
import pandas as pd
from keras.models import load_model

test_data = pd.read_csv('test.csv', sep = ',', encoding = 'UTF-8')
test_data = test_data.as_matrix()
id_data = test_data[:,0]
test_num = len(id_data)
test = list()
for i in range(test_num):
	temp = test_data[i][1].split()
	test.append(temp)
test = np.array(test).astype(int).reshape(test_num,48,48,1)/255.0
model = load_model('model_generator.h5')
result1 = model.predict(test)
del model
model = load_model('model_fit.h5')
result2 = model.predict(test)
del model
model = load_model('model_63.h5')
result3 = model.predict(test)
del model
model = load_model('model_64.h5')
result4 = model.predict(test)
del model
model = load_model('VGG16_model_best.h5')
result5 = model.predict(test)
del model
result = (result1 * 0.1 + result2 * 0.2 + result3 * 0.1 + result4 * 0.2 + result5 * 0.4)
result = result.argmax(axis=-1)

output = pd.DataFrame({'id': id_data , 'label':result})
output.to_csv('Voting.csv', index = False)

