import numpy as np 
import matplotlib.pyplot as plt 
from utilities.blobs import timeit 
from utilities.utilities2 import * #file that contains the loading function of the MNIST set
from tqdm import tqdm #functions that displays the real time progression of the model



#EVERYTHING ABOUT THE COMPUTING METHOD AND SPECIFIC COMPUTATIONS DONE BY THE MODEL ARE WROTE IN A WORD DOCUMENT. 

class Neural:
    """EVERYTHING ABOUT THE COMPUTING METHOD AND SPECIFIC COMPUTATIONS DONE BY THE MODEL ARE WROTE IN A WORD DOCUMENT.
    """
    def __init__(self, X, y, hidden_layers, test=False, XTest=np.ones([1, 1]), yTest=np.ones([1, 1]), learning_rate=0.1, n_iter=600, seed=0) -> None:
        """Creation of the object

        Parameters
        ----------
        X : np.array()
            Size must be (nxm) where n is the number of "variables" for each data and m is the quantity of data.
            X must also be normalized and flatten (using the normFlat() function).
            
            For a pixel set : 
                The correct way of creating the X array is (for a raw set of the size (i, i, m) where i is the first dimension of the squared photo in number of pixels and m is the number of photos):
                "X = normFlat(x_train).T"
        y : np.array()
            The array containing the correct values (for each last neuron) of the last activation array (the array returned by the model). 
            Its size must be (jxm) where j is the number of possible answers and it must contains only 0s and 1s. 
            Example of a y line : [0, 1, 0, 0, 0, 0, 0, 0, 0, 0].
            
            The correct way of creating the y array is (for an set with 10 different possible outputs):
            "y = convertY(y_train, dim=(10, len(y_train)))"

        hidden_layers : tuple
            Tuple containing the number of neurons for each "inside" layer of the models (not for the first layers which is reserved for the number 
            of variables and not for the last one which is reserved for the number of different possible outputs).
        test : bool, optional
            True if the model should compute the accuracy and the log loss on a test set (the test set should be passed in the args), by default False.
        XTest : np.array(), optional
            Same as the X arg, by default np.ones([1, 1])
        yTest : np.array(), optional
            Same as the y arg, by default np.ones([1, 1])
        learning_rate : float, optional
            The factor that multiplies the gradient of the parameter W or b to update it (the more it is set to, the quicker the model will go to the optimum of the function
            but the less it will be accurate because it'll "bounce" on the side of the function when it'll be close to its optimum.) by default 0.1
        n_iter : int, optional
            The number of iteration to do, by default 600.
        seed : int, optional
           The seed of the np.random.randn() function which fills the initialized parameter W and b. 
           Changing the seed means changing the starting point of the model on the optimisation function. By default 0
        """
        self.X = X
        self.y = y
        self.hidden_layers = hidden_layers
        self.test = test
        self.XTest = XTest
        self.yTest = yTest
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.seed = seed
        
        self.param = {}
        self.activations = {}
        
        self.gradients = {}
        
        self.Loss = []
        self.LossTest = []
        self.acc = []
        self.accTest = []
    
    def initialize(self, dimensions:list, seed=0):
        """This functions initializes the parameters W and b with the correct size for each layer of the model.
        It fills them with random numbers created by the np.random.randn() function. The seed is set to 0 by defalut

        Parameters
        ----------
        dimensions : list
            The dimensions of the model. Example : (784, 16, 16, 10) corresponds to a model with 784 entry variables and 10 outputs that 
            contains two layers of both 16 neurons.
        """
        
        np.random.seed(self.seed)
        C = len(dimensions)
        
        for c in range(1, C): #creation des parametres W (n^[c] x n^[c-1]) et b pr chaque couche du reseau (le premier élément et le dernier élément de la liste dimensions sont ignorés grâce aux propriétés de la fonction range())
            self.param['W'+str(c)] = np.random.randn(dimensions[c], dimensions[c-1])
            self.param['b'+str(c)] = np.random.randn(dimensions[c], 1)
    
    def forwardPropagation(self, X): 
        """This function calculates the activation arrays of each layer of the model by using the generated W and b parameters.

        Parameters
        ----------
        X : np.array()
            Size must be (nxm) where n is the number of "variables" for each data and m is the quantity of data.
            X must also be normalized and flatten (using the normFlat() function).
        """
        
        self.activations['A0'] = X
        
        C = len(self.param)//2 #si le reseau fait 3 couches, il aura 2 paramètres (W&b) pr chaque couche donc 2x3//2 = 3 
        
        for c in range(1, C+1): #calcule les activations A pr chaque couche de neurones 
            Z = self.param['W'+str(c)].dot(self.activations['A'+str(c-1)]) + self.param['b'+str(c)]
            self.activations['A'+str(c)] = 1/(1+np.exp(-Z))

    
    def log_loss(self, A, y):
        """This function computes the log loss of the model for an activation array "A" and the array containing the correct
        values for each data under the form of a numpy array with only 0s and 1s.

        Parameters
        ----------
        A : np.array()
            The activation array computed by the model. It's size must be (n^[i] x m) where n^[i] is the number of neurons in the layer i of the model and m is the quantity of data
            contained in the set.
        y : np.array()
            The array containing the correct values (for each last neuron) of the last activation array (the array returned by the model). 
            Its size must be (jxm) where j is the number of possible answers and it must contains only 0s and 1s. 
            Example of a y line : [0, 1, 0, 0, 0, 0, 0, 0, 0, 0].

        Returns
        -------
        float
            The calculated log loss. The lower it is the more accurate the model is.
        """
        epsilon=1e-15 #to prevent from having to compute "log(0)"."
        return 1/y.shape[1] * np.sum(-y* np.log(A+epsilon)-(1-y)*np.log(1-A+epsilon))
    
    def backPropagation(self):
        """This function calculates the gradients via the "back propagation" method which consist of going through the model 
        from the last layer to the first.
        """
        m = self.y.shape[1] 
        C = len(self.param)//2 #pareil que pr forward propagation 
        
        dZ = self.activations['A'+str(C)]-self.y #on calcule d'abord le dZ de la dernière couche
        
        for c in reversed(range(1, C+1)): #on parcourt les couches dans le sens inverse 
            self.gradients['dW'+str(c)] = 1/m * np.dot(dZ, self.activations['A'+str(c-1)].T)   
            self.gradients['db'+str(c)] = 1/m * np.sum(dZ, axis=1, keepdims=True)     
            
            if c > 1 : #pas de sens de calculer dZ^[0] 
                dZ = np.dot(self.param['W'+str(c)].T, dZ)*self.activations['A'+str(c-1)] * (1-self.activations['A'+str(c-1)]) #on calcule ensuite le dZ de la prochaine couche
    
    def update(self):
        """This function updates the parameters W and b via the method of "descente de gradients"
        """

        C = len(self.param)//2 #number layers in the model
        
        for c in range(1, C+1):
            self.param['W'+str(c)] = self.param['W'+str(c)] - self.learning_rate * self.gradients['dW'+str(c)]
            self.param['b'+str(c)] = self.param['b'+str(c)] - self.learning_rate * self.gradients['db'+str(c)]
            
    @timeit
    def start(self):
        """This functions starts the model on the trained set passed in parameters when creating the object.
            It first initializes the arrays W and b with the correct sizes for each layer of the model.
            Then for each iteration : 
            1) It calculates the activations arrays for each layer 
            2) It calculates the gradients dW and db for each layer
            3) It updates the parameters' arrays W and b for each layer
            
            Every 10 iterations, this function also calculates the Log Loss and the accuracy of the model on the train set 
            (also on the test set if one is passed in the args) and then stores them in lists. It also prints the accuracy to 
            watch the model's progression.
        """
        print('\nDimensions de X: {}\nDimensions de y: {}\n'.format(self.X.shape, self.y.shape))         
        
        dimensions = list(self.hidden_layers) 
        dimensions.insert(0, self.X.shape[0]) #insets the number of variables of the set as the first layer
        dimensions.append(self.y.shape[0]) #appends the number of possible outputs for the set as the last layer
        
        self.initialize(dimensions) #creation du dictionnaire des parametres W & b pr chaque couche
        
        for i in tqdm(range(self.n_iter)):
            self.forwardPropagation(self.X) #creation du dictionnaire des activations A pr chaque couche
            self.backPropagation() #creation des gradients dW et db pr chaque couche 
            self.update() #mise à jour des parametres W & b grâce aux gradients correspondants 
            
            if i%10 == 0:
                C = len(self.param)//2
                self.Loss.append(self.log_loss(self.activations['A'+str(C)], self.y))
                self.acc.append(self.accuracy())
                
                if self.test: #si on test le modele sur un test set
                    self.forwardPropagation(self.XTest)
                    self.LossTest.append(self.log_loss(self.activations['A'+str(C)], self.yTest))
                    self.accTest.append(self.accuracy(test=self.test, x=self.XTest, y=self.yTest))
            
            if i%100 == 0:
                print('\n\nAccuracy: {}\n'.format(self.acc[i//10]))
                

    def plotTrains(self):
        """This functions plots the generated "Loss", "LossTest", "Acc", "AccTest" list in functions of iterations of the model.
        """
        if self.test: 
            _, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 4))
            ax[0].plot(self.Loss, label='Train Loss')
            ax[0].plot(self.LossTest, label='Test Loss')
            ax[0].legend()
            
            ax[1].plot(self.acc, label='Train Accuracy')
            ax[1].plot(self.accTest, label='Test Accuracy')
            ax[1].legend()
            plt.show()
        else:
            _, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 4))
            ax[0].plot(self.Loss, label='Train Loss')
            ax[0].legend()
            
            ax[1].plot(self.acc, label='Train Accuracy')
            ax[1].legend()
            plt.show()
            
    def getParam(self):
        """Returns the generated parameters (W and b for each layer)

        Returns
        -------
        dict
            dictionary containing the W and b arrays for each layer of the model 
        """
        return self.param
    
    def predict(self, X):
        """This functions returns the predicted values from the activations of the last layer of the neural network calculated by the model.

        Parameters
        ----------
        X : The set to compute
            Size must be (nxm) where n is the number of "variables" for each data and m is the quantity of data.
            X must also be normalized and flatten (using the normFlat() function).

        Returns
        -------
        np.array()
            The array containing only True and False values for each data : [True False False FalseTrue False False False False False] = 1 data of m.
            If there is an error in the activations, this functions doesn't corrects it so the returned array may contains these type of lines : 
            - [True False False False True False True False False True] : 3 "True" values instead of 1
            - [False False False False False False False False False False] : 0 "True" values instead of 1
        """
        C = len(self.param)//2
        self.forwardPropagation(X)
        return self.activations['A'+str(C)]>=0.5
    
    def accuracy(self, test=False, x=None, y=None):
        """Returns the accuracy of the neural network either on the train set or the test set

        Parameters
        ----------
        test : bool, optional
            True means that the accuracy is calculated on the test set, False means it is calculated on the train set, by default False.
        x : test set, optional
            Test set that's used to calculate the accuracy of the model. Set must be normalized and flatten (using the normFlat() function) and its dimensions must be 
            (nxm) where n is the number of "variables" for each data and m is the quantity of data. By default, None.
        y : np.array(), optional
            The array containing the correct values (for each last neuron) of the last activation array (the array returned by the model). 
            Its size must be (jxm) where j is the number of possible answers and it must contains only 0s and 1s. By default None.

        Returns
        -------
        float   
            The accuracy on the model on the set. The accuracy is in % and it's calculated by additioning all the correct
            responses of the model on the data and divide them by the number of datas and then turned into %.
        """
        cnt = 0
        acc = 0
        if test:
            y_predict = self.predict(x) 
            for i in range(y.shape[1]):
                cnt+=list(y_predict[:, i])==list(y[:, i]) #because the whole line must be correct [0, 1, 0, 0, 0, 0, 0, 0, 0, 0] == [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
            acc = cnt/(y.shape[1]) * 100
        else:
            y_predict = self.predict(self.X)
            for i in range(self.y.shape[1]):
                cnt+=list(y_predict[:, i])==list(self.y[:, i])
            acc = cnt/(self.y.shape[1]) * 100
            
        return acc 
    
            
    def trust(self, X, y):
        """This function returns a degree (=number) of trust for each value of the predicted set.   

        Parameters
        ----------
        X : np.array()
            Train set to predict. Size must be (n x m) where n is the number of "variables" and m is the quantity of data
        y : np.array()
            Array that contains the correct answer for each data of the train set (X). Size must be (j x m) where j is the number of different answer possible (1 for each neuron)
            and m is the quantity of data (same than for X). The array must contains only 1s and 0s for each neuron.

        Returns
        -------
        list
            list of the trust values for the "m" datas. The values are obtained by additionning the "distances" between the correct value 
            and the probability returned by the neural network (the activation array). Each value is positive and belongs to the interval [0, 100].
            The more the value is low the less the error for the concerned predicted data.
        """
        self.forwardPropagation(X) #updating the activations dictionary for the train set passed in parameter (with the W and b parameters that were already calculated)
        C = len(self.param)//2
        A = self.activations['A'+str(C)]
        T = []
        for i in range(A.shape[1]):
            cnt = 0
            for j in range(A.shape[0]):
                if y[j, i] == 1:
                    cnt+=1 - A[j, i]
                else : 
                    cnt+=A[j, i]
            T.append(cnt/y.shape[0]*100) 
        return T
    
#---------------- outside object functions -----------------------           

x_train, y_train, x_test, y_test = load_data() #utilities2 file

def normFlat(ESet):
    """This function normalizes and flattens a raw set of squared picture in order for the neural network to process it.

    Parameters
    ----------
    ESet : np.array()
        Its size must be (i x i x m) where (i x i) is the size of each picture in pixels and m is the number of pictures.

    Returns
    -------
    np.array()
        Normalized and flatten array, its dimensions are (n x m) where n = i x i and m is the numbers of picture.
        Each value of the returned array belongs to the interval [0, 1]
    """
    normSet = ESet/255 #normalizing each value of the array so they belong to the interval [0, 1]
    
    OSet =np.ndarray([ESet.shape[0], ESet.shape[1]*ESet.shape[2]])
    for i in range(ESet.shape[0]):
        OSet[i]=normSet[i].flatten()
    return OSet


def convertY(y, _=True, dim=(10, 10000)):
    """_= True : This function converts a list containing the correct values for a set to a numpy array
    that contains value of 1 or 0 for the neurons
    _=False: This function converts a numpy array of 1 and 0 (or True and False) to an array containing one value 
    for each data of the training set
    

    Parameters
    ----------
    y : np.array() or list
        The array to convert. It must contains only 1 and 0 (or True and False)
        The dimensions of the arrays must be (jxm) where j is the number of "variables" and m the number of datas.
    _ : bool, optional
        If False, it converts 1 and 0 array to list containing 1 value for each data of the training set. 
        If True, it converts a list of values for a training set to a numpy array containing only 1 and 0 for each neuron.
    dim : tuple, optional 
        If _=True, it's the dimensions of the converted array.


    Returns
    -------
    np.array() (_=True) or list (_=False)
        If _=True : it returns a np.array()
        if _=False : it returns a list. If there's more than one "1" value (ligns of y) by data (column of y) or less than one "1" value, 
        the converted value in the list will be set as "E", meaning that there was an error in the conversion.
    """
    if _:
        conv = np.random.randn(dim[0], dim[1])
        for elem, j in zip(y, range(len(y))) : 
            for i in range(conv.shape[0]): 
                if i == elem : 
                    conv[i, j] = 1
                else : 
                    conv[i, j] = 0
                
    else:
        conv = []
        for i in range(y.shape[1]):
            c = 0
            for j in range(y.shape[0]):
                if y[j, i] == 1:
                    mem = j
                    c+=1
                else:
                    pass
            if c>1:
                conv.append('E')
            elif c==0:
                conv.append('E')
            else: 
                conv.append(mem)
    return conv

def plotNmbrs(xSet, p, ySet, T_list):
    """This functions plot the number of the MNIST train set or test set with the predicted value of each number from the neural 
       network, the real value and the trust value for each prediction from the neural network

    Parameters
    ----------
    xSet : np.array()
        Numpy array that must have a size (i x i x n) where i is the number of pixels of one dimensions from the squared picture
        and n is the number of pictures.    
    p : list
        list containing the predicted values given by the neural network for each number of the xSet. 
    ySet : list
        list containing the correct values for each number of the xSet.
    T_list : list
        List containing a positive value for each predicted value from the neural network. The more the value is low 
        the more "sure" the algorithm was about the prediction
    """
    plt.figure(figsize=(16,8))
    for i in range(1, 20):   #because plt cannot display more than 20 pictures 
        plt.subplot(4, 5, i)
        plt.imshow(xSet[i+15], cmap='gray')
        plt.title('[Pred: {}; Set: {}; Trust: {}]'.format(p[i+15], ySet[i+15], round(T_list[i+15], 1)))
        p
        plt.tight_layout()
    plt.show()
    
#-------------------------------- Execution part --------------------------------------

N1 = Neural(normFlat(x_train).T, convertY(y_train, dim=(10, len(y_train))), (8, 16, 8), test=True, XTest=normFlat(x_test).T, yTest=convertY(y_test, dim=(10, len(y_test))), n_iter=500, learning_rate=1.9)
N1.start()
N1.plotTrains()
print('Accuracy: {}'.format(N1.accuracy()))
pred = N1.predict(normFlat(x_test).T)
T_list = N1.trust(normFlat(x_test).T, convertY(y_test, dim=(10, len(y_test))))

plotNmbrs(x_test, convertY(pred, _=False), y_test, T_list)






