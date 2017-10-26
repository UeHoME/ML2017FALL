import numpy as np
import math
from numpy.linalg import det
from numpy.linalg import inv

def get_data(fname):
    text = open(fname,'r')
    text.readline()
    x = []
    for r in text:
        r = r.split('\n') 
        r = r[0].split(',')
        for i in range(len(r)):
            r[i] = float(r[i])
        x.append(r)
    x = np.array(x)
    text.close()
    return x


def min_max_normalize(data):
    row,col = data.shape
    need_normal = [0, 2, 3, 4]
    min_list = []
    max_list = []
    for i in need_normal:
        minimum = np.min(data[:, i], axis=0)
        maximum = np.max(data[:, i], axis=0)
        min_list.append(minimum)
        max_list.append(maximum)
        for j in range(row):
                data[j, i] = (data[j,i] - minimum) / (maximum - minimum)      
    np.savetxt('min.txt', min_list)
    np.savetxt('max.txt', max_list)
    return data

def sigmoid(z):
    try:  
        size,weight = z.shape
    except:
        size = 1
    sig = np.zeros((size,1))
    sig = 1 / (1 + np.exp(-z))
    return sig

def grad_reg(y,x,w1,w2,b):
    weight_decay = 1e-4
    double_x = x ** 2
    size, weight = x.shape
    grad_w1 = np.zeros((weight, 1))
    grad_w2 = np.zeros((weight, 1))
    z = np.dot(x, w1) + np.dot(double_x, w2) + b
    grad = y - sigmoid(z)
    grad_w1 = (sum((grad.T * x.T).T ).reshape(weight,1) + 2 * weight_decay * w1).reshape(weight,1)
    grad_w2 = (sum((grad.T * double_x.T).T).reshape(weight,1) + (2 * weight_decay * w2.T).reshape(weight,1)).reshape(weight,1)

    return (- 1 * (grad_w1 )), (- 1 * (grad_w2 ) ) , (- 1 * (sum(grad)))

def grad(y,x,w1,w2,b):
    double_x = x ** 2
    size, weight = x.shape
    grad_w1 = np.zeros((weight, 1))
    grad_w2 = np.zeros((weight, 1))
    z = np.dot(x, w1) + np.dot(double_x, w2) + b
    grad = y - sigmoid(z)
    grad_w1 = sum((grad.T * x.T).T).reshape(weight,1)
    grad_w2 = sum((grad.T * double_x.T).T).reshape(weight,1)

    return (- 1 * (grad_w1 )), (- 1 * (grad_w2) ) , (- 1 * (sum(grad)))

def cut_validation(y, x):
    data, weight = x.shape
    train_num = int(data * 0.9)
    val_num = data - train_num
    idx = np.arange(x.shape[0])
    np.random.shuffle(idx)
    y = y[idx]
    x = x[idx]
    val_label = y[0:val_num,]
    val_data  = x[0:val_num, ] 
    train_label = y[val_num:,]
    train_data  = x[val_num:,]

    return train_label, train_data, val_label, val_data

def do_shuffle(y, x):
    idx = np.arange(x.shape[0])
    np.random.shuffle(idx)

    return y[idx], x[idx]



def Adam(grad_w1,w1,grad_w2,w2,grad_b,b,lr):
    (weight,size) = w1.shape

    m_w1 = np.zeros((weight, 1))
    v_w1 = 0
    m_w2 = np.zeros((weight, 1))
    v_w2 = 0
    m_b  = 0
    v_b  = 0
    time = 0
    Beta1 = 0.9
    Beta2 = 0.999
    epilson = 1e-5
    m_w1 = Beta1 * m_w1 + (1 - Beta1) * grad_w1
    v_w1 = Beta2 * v_w1 + (1 - Beta2) * np.dot(grad_w1. T, grad_w1)
    m_w1_update = m_w1 / (1 - Beta1 )
    v_w1_update = v_w1 / (1 - Beta2 )

    m_w2 = Beta1 * m_w2 + (1 - Beta1) * grad_w2
    v_w2 = Beta2 * v_w2 + (1 - Beta2) * np.dot(grad_w2. T, grad_w2)
    m_w2_update = m_w2 / (1 - Beta1 )
    v_w2_update = v_w2 / (1 - Beta2 )


    m_b = Beta1 * m_b + (1 - Beta1) * grad_b
    v_b = Beta2 * v_b + (1 - Beta2) * (grad_b ** 2)
    m_b_update = m_b / (1 - Beta1 )
    v_b_update = v_b / (1 - Beta2)
    
    w1 = w1 - ( lr  * m_w1_update) / (math.sqrt(v_w1_update) + epilson)
    w2 = w2 - ( lr  * m_w2_update) / (math.sqrt(v_w2_update) + epilson)
    b = b - (lr * m_b_update ) / (math.sqrt(v_b_update[0]) +  epilson)

    return w1,w2,b

def mini_batch(y, x, size):
    y, x = do_shuffle(y, x)
    i= 0
    n = 0
    num, weight = x.shape
    while n < num:
        data = x[n:size + n,:]
        label = y[n:size + n,:]
        i += 1
        yield label, data
        n += size
        
    data = x[n:num,:]
    label = y[n:num,:]
    yield label, data

def accuracy(y,x,w1,w2,b):
    acc = 0.0
    double_x = x ** 2
    size,weight = x.shape
    z = np.dot(x,w1) + b + np.dot(double_x, w2)
    grad = y - (sigmoid(z))
    acc = np.sum(abs(grad) <= 0.5)
    return acc / float(size)

def validation(y,x,w1,w2,b):
    val = 0.0
    double_x = x ** 2
    size,weight = x.shape
    z = np.dot(x,w1) + b + np.dot(double_x, w2)
    grad = y - (sigmoid(z))
    val = np.sum(abs(grad) <= 0.5)
    return val / float(size)

def focal_loss(Ground_truth,predict,a,r):
    size,weight = Ground_truth.shape
    grad = 0 
    for i in range(size):
        y = Ground_truth[i]
        p = predict[i]
        if y == 1:
            grad += abs((-a * (p ** 2)) * math.log(p))
        else:
            grad += abs(((1 - a) * ((1 - p) ** 2)) * math.log(p))
    return grad



def cross_multipy(target,axis = None):
    size,weight = target.shape
    product = []
    if axis == 0 or axis == None:
        for j in range(weight):
            for i in range(size):
                for k in range(i,size):

                    cross = target[i, j] * target[k, j]
                    product.append(cross)
                if j == 0 and i == size - 1:    
                    length = len(product)
        product = np.array(product).reshape(length, weight) 

    elif axis == 1:
        for j in range(size):
            for i in range(weight):
                for k in range(weight):
                    cross = target[j, i] * target[j, k]
                    product.append(cross)
                if j == 0 and i == size - 1: 
                    length = len(product)
        product = np.array(product).reshape(size, length)                
    return product

class Generative_model:
    def __init__(self):
        self.x = None

    def Maximum_Likelihood(self,x):
        self.x = x
        self.size,self.weight = self.x.shape
        self.means = np.zeros((self.weight,1))
        self.covar = np.zeros((self.weight, self.weight))
        self.means = sum(x) / self.size
        self.covar = (np.dot((x - self.means).T,(x - self.means))) / self.size
    def Gaussian_Distrobution(self,x):
        pi = math.pi
        cons1 = (1 / ((2 * pi) ** (self.weight / 2))) * (1 / (det(self.covar) ** (1 / 2)))
        cons2 = np.exp((- 1 / 2) * np.dot((x - self.means),np.dot(inv(self.covar),(x - self.means))))
        return cons1 * cons2
        
def classification(y,x):
    size,weight = x.shape
    label_1 = []
    label_0 = []
    for i in range(size):
        if y[i] == 0 :
            label_0.append(i)
        else: 
            label_1.append(i)

    class1 = np.zeros((len(label_1),weight))
    class0 = np.zeros((len(label_0),weight))
    n = 0
    for i in label_0:
        class0[n,:] = x[i,:]
        n += 1
    n = 0
    for i in label_1:
        class1[n,:] = x[i,:]
        n += 1  
    return class0,class1