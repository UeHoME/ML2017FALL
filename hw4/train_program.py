import pandas as pd
import numpy as np
import pickle
from keras.preprocessing import sequence, text
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import GRU, Dense, Activation, Dropout
from keras.layers.normalization import BatchNormalization
import matplotlib.pyplot as plt
# parameters
max_word_idx = 5000
max_sequence_len = 50
nb_val = 200000
embedding_vector_len = 50
threshold = 0.5
# get labeled training data



data_labeled = pd.read_csv(sys.argv[1], sep = '\+\+\+\$\+\+\+', header = None, engine = 'python')
x_labeled = data_labeled[1].tolist()
y_labeled = data_labeled[0].tolist()

x_unlabeled = [line.rstrip('\n') for line in open(sys.argv[2], 'r')]

t = text.Tokenizer(num_words = max_word_idx)
t.fit_on_texts(x_labeled + x_unlabeled)
# with open('tokenizer.pickle', 'wb') as handle:
#     pickle.dump(t, handle)

x_labeled = t.texts_to_sequences(x_labeled)
x_unlabeled = t.texts_to_sequences(x_unlabeled)

x_labeled = sequence.pad_sequences(x_labeled, maxlen = max_sequence_len)
x_unlabeled = sequence.pad_sequences(x_unlabeled, maxlen = max_sequence_len)
x_labeled = np.asarray(x_labeled)
y_labeled = np.asarray(y_labeled).reshape(-1,1)
x_unlabeled = np.asarray(x_unlabeled)

model = Sequential()
model.add(Embedding(max_word_idx, embedding_vector_len, input_length = max_sequence_len))
model.add(GRU(10))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.summary()

# training
history = model.fit(x_labeled, y_labeled, nb_epoch = 4, batch_size = 5000,  validation_split = 0.1,)
# model.save('model_com.h5')

print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
