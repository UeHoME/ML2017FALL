import numpy as np
import sys
import function

data = function.get_data(sys.argv[3])
label = function.get_data(sys.argv[4])

row ,col = data.shape

training_data = np.zeros((row, col - 1 ))
n = 0
for i in range(col):
    if i != 1:
        training_data[:, n] = data[:, i]
        n += 1

row ,col = training_data.shape

training_data = function.min_max_normalize(training_data)

train_label, train_data, val_label, val_data = function.cut_validation(label, training_data)
train_num,weigth = train_data.shape
w1 = np.random.normal(0.0, 0.1, (col, 1))
w2 = np.random.normal(0.0, 0.1, (5, 1))
b = 0
lr = 0.002
for epoch in range(101):    
    if epoch % 1000 == 0:
        lr = lr / 2
    batches = function.mini_batch(train_label,train_data, batch_size)
    for y, x in batches:
        data,weight = x.shape
        grad_w1, grad_w2, grad_b = function.grad(y,x,w1,w2,b)
        w1,w2,b = function.Adam(grad_w1,w1,grad_w2,w2,grad_b,b,lr)
        if data != batch_size:
            break

    accur = function.accuracy(train_label, train_data, w1, w2, b)
    val_accur = function.validation(val_label, val_data, w1, w2, b)

    sys.stdout.write('\r{} {} {} {} {} {:7.5f} {} {:7.5f}'.format('epoch=',epoch,'lr=',lr, 'train accuracy=',accur ,'val accuracy=',val_accur))
    sys.stdout.flush()
print("")
