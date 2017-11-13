import numpy as np
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten, Dropout
from keras.models import Sequential
from keras.optimizers import Adam,SGD
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers
from sklearn.metrics import confusion_matrix
import sys
text = open(sys.argv[1])
text.readline()
x = []
y = []
a = 0
for r in text:
    a += 1
    label = r[0]
    r = r[2:].split(',')
    r = r[0].split(' ')
    x.append(r)
    y.append(label)
text.close()
train = np.array(x).reshape(-1,48,48,1).astype(float)
img = np.zeros((48,48))
train = train / 256.0
label = keras.utils.to_categorical(np.array(y), num_classes = 7)



model = Sequential()
# layer 1

model.add(Convolution2D(filters = 32,kernel_size = (3,3),padding ="same",input_shape = (48,48,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Convolution2D(filters = 32,kernel_size = (3,3),padding ="same"))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2),strides = (2, 2)))
#layer 2 input 64x24x24

model.add(Convolution2D(filters = 64,kernel_size = (3,3),padding ="same"))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Convolution2D(filters = 64,kernel_size = (3,3),padding ="same"))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2),strides = (2, 2)))
#layer 3 input 128x12x12

model.add(Convolution2D(filters = 128,kernel_size = (3,3),padding ="same",))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Convolution2D(filters = 128,kernel_size = (3,3),padding ="same"))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Convolution2D(filters = 128,kernel_size = (3,3),padding ="same"))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2),strides = (2, 2)))
 
model.add(Convolution2D(filters = 256,kernel_size = (3,3),padding ="same"))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Convolution2D(filters = 256,kernel_size = (3,3),padding ="same"))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Convolution2D(filters = 256,kernel_size = (3,3),padding ="same"))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2),strides = (2, 2)))


model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(384,activation = 'relu', kernel_regularizer = regularizers.l2(0.01)))
model.add(Dropout(0.5,))
model.add(Dense(384,activation = 'relu'))
model.add(Dropout(0.5, kernel_regularizer = regularizers.l2(0.01)))
model.add(Dense(7,activation = 'softmax'))
model.summary()

train_generator = ImageDataGenerator(rotation_range = 30, horizontal_flip=True, width_shift_range = 0.2, height_shift_range = 0.2, )
val_generator = ImageDataGenerator()
train_generator.fit(train[:25000])
val_generator.fit(train[25000:])
nb =len(train[:25000]) / 256
# adam = Adam(lr = 3e-4,decay = 0.1)
sgd = SGD(lr=0.01, momentum=0.9, decay=5e-4)
checkpoint1 = ModelCheckpoint('model_r06921066.h5',monitor = 'val_acc',save_best_only = True)
# checkpoint2 = ModelCheckpoint('model_fit.h5',monitor = 'val_acc',save_best_only = True)
# earlystop = EarlyStopping(monitor = 'val_acc',patience=10

model.compile(optimizer = sgd, loss = 'categorical_crossentropy', metrics = ['accuracy'])

model.fit_generator(train_generator.flow(train[:25000],label[:25000],batch_size = 256),
    steps_per_epoch = nb,
    epochs = 200,
    validation_data = val_generator.flow(train[25000:],label[25000:],batch_size = 256),
    validation_steps = len(train[25000:]) / 256,
    callbacks =[checkpoint1],)
# model.fit(train, label, epochs = 50,
#         batch_size = 256, validation_split = 0.1,
#         callbacks = [checkpoint2], shuffle = True)