from gensim.models import  word2vec
import pandas as pd
import numpy as np
import jieba
import sys
embedding_dimention = 64;

def cos_sim(u, v):
    return np.dot(u,v) / (np.linalg.norm(u) * np.linalg.norm(v))



# set jieba dictionary
jieba.set_dictionary('jieba_dict/dict.txt.big.txt')

# load stopwords set
stopwordset = set()
with open('jieba_dict/stop_words.txt', 'r', encoding = 'utf-8') as sw:
    for line in sw:
        stopwordset.add(line.strip('\n'))

# load model
model = word2vec.Word2Vec.load("model_64.bin")

# load data
data = pd.read_csv(sys.argv[1])

# do stupid prediction
result = []
for idx in range(5060):
    # processing on question
    question = jieba.cut(data.iloc[idx]['dialogue'], cut_all = False)
    question_seg = []
    nb_word = 0
    question_vec = np.zeros((embedding_dimention,))
    for word in question:
        if word not in stopwordset:
            try:
                nb_word += 1
                question_vec += model[word]
            except:
                pass
    question_vec = question_vec / nb_word

    # processing on answer
    answers = data.iloc[idx]['options'].split("\t")
    score = []


    choice_0 = jieba.cut(answers[0], cut_all = False)
    choice_0_seg = []
    option_vec = np.zeros((embedding_dimention,))
    nb_word = 0
    for word in choice_0:
        if word not in stopwordset:
            try:
                option_vec += model[word]
                nb_word += 1
            except:
                pass
    option_vec = option_vec / nb_word
    score.append(cos_sim(question_vec,option_vec))

        # print(score)
    choice_1 = jieba.cut(answers[1], cut_all = False)
    choice_1_seg = []
    option_vec = np.zeros((embedding_dimention,))
    nb_word = 0
    for word in choice_1:
        if word not in stopwordset:
            try:
                option_vec += model[word]
                nb_word += 1
            except:
                pass
    option_vec = option_vec / nb_word
    score.append(cos_sim(question_vec,option_vec))

    choice_2 = jieba.cut(answers[2], cut_all = False)
    choice_2_seg = []
    option_vec = np.zeros((embedding_dimention,))
    nb_word = 0
    for word in choice_2:
        if word not in stopwordset:
            try:
                option_vec += model[word]
                nb_word += 1
            except:
                pass
    option_vec = option_vec / nb_word
    score.append(cos_sim(question_vec,option_vec))

    choice_3 = jieba.cut(answers[3], cut_all = False)
    choice_3_seg = []
    option_vec = np.zeros((embedding_dimention,))
    nb_word = 0
    for word in choice_3:
        if word not in stopwordset:
            try:
                option_vec += model[word]
                nb_word += 1
            except:
                pass
    option_vec = option_vec / nb_word
    score.append(cos_sim(question_vec,option_vec))

    choice_4 = jieba.cut(answers[4], cut_all = False)
    choice_4_seg = []
    option_vec = np.zeros((embedding_dimention,))
    nb_word = 0
    for word in choice_4:
        if word not in stopwordset:
            try:
                option_vec += model[word]
                nb_word += 1
            except:
                pass
    option_vec = option_vec / nb_word
    score.append(cos_sim(question_vec,option_vec))

    choice_5 = jieba.cut(answers[5], cut_all = False)
    choice_5_seg = []
    option_vec = np.zeros((embedding_dimention,))
    nb_word = 0
    for word in choice_5:
        if word not in stopwordset:
            try:
                option_vec += model[word]
                nb_word += 1
            except:
                pass
    option_vec = option_vec / nb_word
    score.append(cos_sim(question_vec,option_vec))

    final_choice = score.index(max(score))
    result.append(final_choice)
    # print(score, final_choice)

    if idx % 1000 == 0:
        print("%d predictions have been done" % idx)

result = np.asarray(result).reshape(-1,1)
print(result.shape)

# generate prediction file
id_col = np.array([str(i+1) for i in range(result.shape[0])]).reshape(-1,1)
output = np.hstack((id_col, result.astype(int)))
output_df = pd.DataFrame(data = output, columns = ['id', 'ans'])
output_df.to_csv(sys.argv[2], index = False)
