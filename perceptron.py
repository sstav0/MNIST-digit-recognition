import numpy as np 
import matplotlib.pyplot as plt 
from utilities.blobs import timeit 
from utilities.utilities import *
from tqdm import tqdm

x_train, y_train, x_test, y_test = load_data()
print(x_train.shape, y_train.shape)

def plotCats(xSet, ySet):
    plt.figure(figsize=(16,8))
    for i in range(1, 20):
        plt.subplot(4, 5, i)
        plt.imshow(xSet[i], cmap='gray')
        plt.title(ySet[i])
        plt.tight_layout()
    plt.show()

def normFlat(ESet):
    normSet = ESet/255
    #print("ESET SHAPE: {}".format(ESet.shape))
    OSet =np.ndarray([ESet.shape[0], ESet.shape[1]*ESet.shape[2]])
    for i in range(ESet.shape[0]):
        OSet[i]=normSet[i].flatten()
    return OSet
    

# def fun1(x:int):
#     return x**4+x**3+x**2+x+0.08 

# X = blobs.random_array(500, 2)
# y = blobs.YValue(X, fun1)

class Perceptron:
     
    def __init__(self, X, y, XTest, yTest, learning_rate=0.1, n_iter=100) -> None:
        self.X = X
        self.y = y
        self.XTest = XTest
        self.yTest = yTest
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        
        self.W = np.ones([2,2])
        self.b = 0
        self.Loss = []
        self.LossTest = []
        self.acc = []
        self.accTest = []
    
    def initialize(self, X):
        W = np.random.randn(X.shape[1], 1)
        b = np.random.randn(1)
        return (W, b)
    
    def model(self, X, W, b): 
        Z = X.dot(W)+b
        A = 1/(1+np.exp(-Z))
        return A
    
    def log_loss(self, A, y):
        epsilon=1e-15
        return 1/len(y) * np.sum(-y* np.log(A+epsilon)-(1-y)*np.log(1-A+epsilon))
    
    def gradients(self, A, X, y):
        dW = 1/len(y) * np.dot(X.T, A-y)
        db = 1/len(y) * np.sum(A-y)
        return(dW, db)
    
    def update(self, dW, db, W, b, learning_rate):
        W = W - learning_rate * dW
        b = b - learning_rate * db
        return(W,b)
    
    @timeit
    def artificial_neuron(self):
        self.W, self.b = self.initialize(self.X)
        
        for i in tqdm(range(self.n_iter)):
            A = self.model(self.X, self.W, self.b)
            ATest = self.model(self.XTest, self.W, self.b)
            dW, db = self.gradients(A, self.X, self.y)
            self.W, self.b = self.update(dW, db, self.W, self.b, self.learning_rate)
            
            if i%10 == 0:
                self.Loss.append(self.log_loss(A, self.y))
                self.LossTest.append(self.log_loss(ATest, self.yTest))
                self.acc.append(self.accuracy())
                self.accTest.append(self.accuracy(test=True, x=self.XTest, y=self.yTest))

    def plotLoss(self):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(self.Loss, label='Train Loss')
        plt.plot(self.LossTest, label='Test Loss')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(self.acc, label='Train Accuracy')
        plt.plot(self.accTest, label='Test Accuracy')
        plt.legend()
        plt.show()
    
    def plotTrainset(self):
        print('dimensions de X: ', self.X.shape)
        print('dimensions de Y: ', self.y.shape)

        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.y, cmap='summer')
        plt.show()
        
    def plotFronteer(self):
        x0 = np.linspace(np.amin(self.X), np.amax(self.X), 100)
        x1 = (-self.W[0] * x0 - self.b)/self.W[1]
        
        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.y, cmap='summer')
        plt.plot(x0, x1, c='orange', lw=3)
        plt.show()
        
        
    def getParam(self):
        return (self.W, self.b)
    
    def predict(self, X):
        A = self.model(X, self.W, self.b)
        return A>=0.5
    
    def accuracy(self, test = False, x=None, y=None):
        cnt = 0
        acc = 0
        if test:
            y_predict = self.predict(x)
            for i in range(y.shape[0]):
                cnt+=int(y_predict[i, 0]==y[i, 0])
            acc = cnt/y.shape[0] * 100
        else:
            y_predict = self.predict(self.X)
            for i in range(self.y.shape[0]):
                cnt+=int(y_predict[i, 0]==self.y[i, 0])
            acc = cnt/self.y.shape[0] * 100
            
        #print("\nACCURACY: {}\n".format(acc))
        return acc 
    

#-------------------------------- Execution part --------------------------------------


P = Perceptron(normFlat(x_train), y_train, normFlat(x_test), y_test, learning_rate=0.01, n_iter=20000)
P.artificial_neuron()
print(P.accuracy())
P.plotLoss()
print(P.accuracy(test=True, x=normFlat(x_test), y=y_test)) 
plotCats(x_test, P.predict(normFlat(x_test)))




        
    
