from keras.preprocessing import sequence, text
from keras.models import Sequential,Model
from keras.layers.embeddings import Embedding
from keras.layers import  Dense, Dropout,Input,Flatten,Add
from keras.layers.merge import Dot
from keras.callbacks import  ModelCheckpoint
import keras
import numpy as np
import pandas as pd

def load_data(fileName):
    dataFrame = pd.read_csv(fileName, encoding='utf8')

    user_id = dataFrame['UserID']
    user_id = user_id.apply(pd.to_numeric).as_matrix()-1
    num_users = dataFrame['UserID'].drop_duplicates().max()

    movie_id = dataFrame['MovieID']
    movie_id = movie_id.apply(pd.to_numeric).as_matrix()-1
    m_items = dataFrame['MovieID'].drop_duplicates().max()

    rating = dataFrame['Rating']
    rating = rating.apply(pd.to_numeric).as_matrix()

    return user_id, num_users, movie_id, m_items, rating

def movie_dict(fname):
    with open(fname) as f:
        f.readline()
        for i in f:
            i = i[:].split("::")
            r = i[2].split('\n')
            MOVIE_NB.append(i[0])
            MOVIE_NAME.append(r[0])

def user_dict(fname):
    with open(fname) as f:
        f.readline()
        for i in f:
            i = i[:].split("::")
            if i[1] == 'F':
                i[1] = 0
            else:
                i[1] = 1
            USER_dict[int(i[0])] = [int(i[0]), int(i[1]), int(i[2])]

def shuffle(User_id,movie_id,rating):
    idx = np.random.permutation(User_id.shape[0])
    User_id = User_id[idx]
    movie_id = movie_id[idx]
    rating = rating[idx]
    return User_id, movie_id, rating

# ============= Definition container ======================= # 

USER_dict = {}
USER_ID = []
MOVIE_ID = []
MOVIE_NB = []
MOVIE_NAME = []
RATING = [] 
movie = {}
movie_id = []
user_id = []

# ============= Setting parameters ======================= #  

max_sequence_len = 1
embedding_vector_len = 64
batch_size = 256
epochs = 100

# ============= Only use training ddata ================== #

USER_ID, num_users, MOVIE_ID, m_items, RATING = load_data("./data/train.csv")

# ===============  Rating Normalize ==================== #

# maximum = np.max(RATING)
# minimum = np.min(RATING)
# RATING = RATING.astype(float) 
# for i in range(len(RATING)):
#     RATING[i]  = (RATING[i] - minimum) / (maximum - minimum)

# ========================= shuffle ========================= #

USER_ID, MOVIE_ID, RATING = shuffle(USER_ID, MOVIE_ID, RATING)

# ========================== MF model =========================== #

# user_input = Input(shape = (max_sequence_len,))
# user_vec = Embedding(num_users, embedding_vector_len)(user_input)
# user_vec =  Dropout(0.5)(Flatten()(user_vec))
# movie_input = Input(shape = (max_sequence_len,))
# movie_vec = Embedding(m_items, embedding_vector_len)(movie_input)
# movie_vec =  Dropout(0.5)(Flatten()(movie_vec))
# input_vec = Dot(axes = 1)([user_vec, movie_vec])

# u_bias = Embedding(num_users, 1)(user_input)
# u_bias = (Flatten()(u_bias))
# v_bias = Embedding(m_items, 1)(movie_input)
# v_bias = (Flatten()(v_bias))

# out = Add()([input_vec, u_bias, v_bias])
# model = Model([user_input, movie_input], out)
# model.summary()

# =========================== DNN model ========================= #

user_input = Input(shape = (max_sequence_len,))
user_vec = Embedding(num_users, embedding_vector_len)(user_input)
user_vec =  Dropout(0.5)(Flatten()(user_vec))
movie_input = Input(shape = (max_sequence_len,))
movie_vec = Embedding(m_items, embedding_vector_len)(movie_input)
movie_vec =  Dropout(0.5)(Flatten()(movie_vec))
input_vec = Dot(axes = 1)([user_vec, movie_vec])

u_bias = Embedding(num_users, 1)(user_input)
u_bias = (Flatten()(u_bias))
v_bias = Embedding(m_items, 1)(movie_input)
v_bias = (Flatten()(v_bias))

out = Add()([input_vec, u_bias, v_bias])
nn = Dropout(0.5)(Dense(128)(out))
result = Dense(1)(nn)
model = Model([user_input, movie_input], result)
model.summary()

# ======================== training ==========================#

model.compile(loss = 'mse', optimizer = 'adam')
checkpoint = ModelCheckpoint('model_64.h5',monitor = 'val_loss',save_best_only = True)

# ======================== training with save.    ====================== #

model.fit([USER_ID, MOVIE_ID], RATING, epochs = epochs, batch_size = batch_size, validation_split = 0.1,callbacks = [checkpoint])

# ======================== training without save ====================== #

# model.fit([USER_ID, MOVIE_ID], RATING, epochs = epochs, batch_size = 512, validation_split = 0.1)

# ======================== get embedding =========================== #

# user_emb = np.array(model.layers[2].get_weights()).squeeze()
# movie_emb = np.array(model.layers[3].get_weights()).squeeze()
# np.save('user_emb.npy',user_emb)
# np.save('movie_emb.npy',movie_emb)

