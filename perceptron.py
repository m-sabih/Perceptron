import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def stepFunction(t):
    if t >= 0:
        return 1
    return 0

def prediction(X, W, b):
    a=(np.matmul(X,W)+b)[0]
    return stepFunction(a)

def perceptronStep(X, y, W, b, learn_rate = 0.01):
    for i in range(len(X)):
        Y=prediction(X[i],W,b)
        if Y-y[i]==-1:
            W[0]=W[0]+(X[i][0]*learn_rate)
            W[1]=W[1]+(X[i][1]*learn_rate)
            b=b+learn_rate
        elif Y-y[i]==1:
            W[0]=W[0]-(X[i][0]*learn_rate)
            W[1]=W[1]-(X[i][1]*learn_rate)
            b=b-learn_rate
    return W, b
    
def trainPerceptronAlgorithm(X, y, learn_rate = 0.01, num_epochs = 35):
    W = np.array(np.random.rand(2,1))
    b = np.random.rand(1)[0]+1
    boundary_lines = []
    for i in range(num_epochs):
        W, b = perceptronStep(X, y, W, b, learn_rate)
        boundary_lines.append((-W[0]/W[1], -b/W[1]))
    return boundary_lines


X=[]
data=pd.read_csv('data.csv',skiprows=0,header=None)
df=pd.DataFrame(data)
X=df.iloc[:,[0,1]].values.tolist()
y=df.iloc[:,2].values.tolist()
np.random.seed(42)
for i,row in enumerate(X):
    if(y[i]==1):
        plt.scatter(row[0],row[1],color='b')
    else:
        plt.scatter(row[0],row[1],color='r')
a=trainPerceptronAlgorithm(X,y)
length=len(a)
p1=[row[0] for row in a]
p2=[row[1] for row in a]
norm1 = [(float(i)-min(p1))/(max(p1)-min(p1)) for i in p1]
norm2 = [(float(i)-min(p2))/(max(p2)-min(p2)) for i in p2]
lines = []
val = np.array(np.random.rand(2,1))
for i,value in enumerate(norm1):
    lines.append((norm1[i],norm2[i]))
for k in lines:
    plt.plot(k)
plt.plot(lines[24],color="yellow",linewidth='3')