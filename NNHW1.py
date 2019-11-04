#!/usr/bin/env python
# coding: utf-8

# In[2]:


from random import choice                   #It is a OR problem
from numpy import array, dot, random

activation = lambda x: 0 if x < 0 else 1    #activation function: X>=1則分類為1, X<0則分類為0

training_data = [                           #手動輸入資料, 其意義為(array([X1座標,X2座標,Bias]), 分類目標)
    (array([0,0,1]), 0),
    (array([0,1,1]), 1),
    (array([1,0,1]), 1),
    (array([1,1,1]), 1),
]

w = random.rand(3)                          #以隨機值當作初始權重(w1,w2,w0)
errors = []
lr = 0.29                                   #lr = learning rate
iteration = 100                             #iteration設為100        

for x, _ in training_data:
    result = dot(w, x)
    print("{}: {} -> {}".format(x[:2], result, activation(result)))
 

for i in range(iteration):
    x, expected = choice(training_data)
    result = dot(w, x)
    error = expected - activation(result)
    errors.append(error)
    w += lr * error * x
    print("iteration %s = %s" % (i+1, w))   #列出每個 iteration所求得的 W值
print(w)                                    #目前的最佳解 W (w1,w2,w0)
                               


from pylab import plot, ylim                #以圖表方式呈現在各個 iteration中,error的值
ylim([-1,1])
plot(errors)

