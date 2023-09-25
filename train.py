#!/usr/bin/env python
# coding: utf-8

# In[79]:


import wandb
import numpy  as np
import pandas as pd
import tensorflow as tf
from keras.datasets import fashion_mnist
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import math
import seaborn as sns
from sklearn.metrics import confusion_matrix
import copy
import argparse


# In[80]:


wandb.login()


# In[81]:


'''LOADING TRAIN AND TEST DATA SET'''
(X_train,Y_train),(X_test,Y_test) = fashion_mnist.load_data()
(X_train_mnist,Y_train_mnist),(X_test_mnist,Y_test_mnist) = mnist.load_data()


# In[82]:


def pre_processing_data(X_train,Y_train,X_test,Y_test):
    
    '''CHANGING THE SHAPE OF INPUT DATA'''
    x_train=np.zeros((60000,784))
    for i in range(X_train.shape[0]):
        a=X_train[i].reshape(1,784)
        x_train[i]=a
        x_test=np.zeros((10000,784))
    for i in range(X_test.shape[0]):
        a=X_test[i].reshape(1,784)
        x_test[i]=a
    '''CONVERTING OUTPUT DATA INTO ONE HOT VECTOR FORM'''
    a = np.max(Y_train)+1
    y_train=np.zeros((Y_train.shape[0],a))
    for i in range(Y_train.shape[0]):
        for j in range(a):
            if Y_train[i]==j:
                y_train[i,j]=1
    y_test=np.zeros((Y_test.shape[0],a))
    for i in range(Y_test.shape[0]):
        for j in range(a):
            if Y_test[i]==j:
                y_test[i,j]=1
    '''CREATING VALIDATION DATA SET'''
    x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,test_size=0.10,random_state=10)
    
    '''Normalisation of data'''
    x_train=x_train.T/255
    y_train=y_train.T
    x_test=x_test.T/255
    y_test=y_test.T
    x_val=x_val.T/255
    y_val=y_val.T
    return x_train,y_train,x_val,y_val,x_test,y_test
    


# In[83]:


def data_augmentation(x):
    '''DATA AUGMENTATION BY ADDING NOISE TO DATA'''
    n,m=x.shape
    mean=0
    variance=0.001
    std=np.sqrt(variance)
    k=np.random.normal(loc=mean,scale=std,size=(n,m))
    x=x+k
    x=np.clip(x,0,1)
    return x


# In[84]:


'''ACTIVATION FUNCTIONS'''

'''input: zl = w(l)*a(l-1) + b(l) where l is the lth Layer.The various activation functions implemented here are 
sigmoid,tanh,ReLu and Identity functions.'''


#SIGMOID FUNCTION
def sigmoid_function(z):
    h=1./(1.+np.exp(-z))
    
    return h

#TANH FUNCTION
def tanh_function(z):
    h=np.tanh(z)
    
    return h

#RELU FUNCTION
def relu_function(z):
    h=np.maximum(z,0)
    
    return h
    

#IDENTITY FUNCTION
def identity_function(z):
    
    return z


# In[85]:


#OUTPUT LAYER FUNCTION
'''The given problem is a multi-class classification problem.So,we use softmax function for the output layer(L)
    Z(L) = W(L)*A(L-1) + B(L) where Lth layer is the output layer.'''


#SOFTMAX FUNCTION
'''OUTPUT LAYER FUNCTION'''

def softmax_function(z):
    #z=z-np.max(z,axis=0,keepdims=True)
    h = np.exp(z)/np.sum(np.exp(z), axis=0)
    return h


# In[86]:


'''DERIVATIVE FUNCTIONS'''

'''These are the derivatives of the corresponding activation functions, which is used in backpropagation to find
the derivative of activation functions.'''

#DERIVATIVE OF SIGMOID FUNCTION
def sigmoid_function_dash(z):
    h = sigmoid_function(z)
    
    return h*(1-h)


#DERIVATIVE OF TANH FUNCTION
def tanh_function_dash(z):
    h=tanh_function(z)
    
    return 1-(h)**2


#DERIVATIVE OF RELU FUNCTION
def relu_function_dash(z):
    return 1*(z>0)
    
#DERIVATIVE OF IDENTITY FUNCTION
def identity_function_dash(z):
    h = identity_function(z)
    
    return np.ones(z.shape)


# In[87]:


'''SOFTMAX DERIVATIVE'''
def softmax_dash(Z):
    h= softmax(z) * (1-softmax(z))
    return h


# In[88]:


#CROSS ENTROPY FUNCTION(DERIVATIVE OF OUTPUT LAYER)
def cross_entropy_function(y,ycap,w,lambd):
    '''This function is called as categorical cross entropy function.
       input: Y:actual value of output
              YCAP:predicted value of output
              lambd:Regularisation parameter(L2 Rregularization is used here)'''
    
    ycap = np.clip(ycap, 1e-12, 1.0 - 1e-12)
    m=y.shape[1]
    cost=-(1/m)*np.sum(y*np.log(ycap))
    regularization_cost=0
    for i in range(len(w)):
        regularization_cost += (lambd/(2*m))*np.sum(np.square(w[i]))
        
    return cost+regularization_cost
     
        
#MEAN SQUARED ERROR FUNCTION
def mean_squared_error_function(y,ycap,w,lambd):
    '''input: Y:actual value of output
              YCAP:predicted value of output
              lambd:Regularisation parameter(L2 Rregularization is used here)'''   
    ycap = np.clip(ycap, 1e-12, 1.0 - 1e-12)
    m = y.shape[1]
    mean_square_error = (1/m)*np.sum((y-ycap)**2)
    reg_cost=0
    for i in range(len(w)):
        reg_cost += (lambd/(2*m))*np.sum(w[i]**2)
    return mean_square_error + reg_cost


# In[89]:


#INITIALISE PARAMETERS

'''input:  Layer_attributes is a list consisting of number of 
    neurons in each layer. Here,input layer is considered as 0th Layer, output layer is considered as Lth layer
    and the layers from 1 to (L-1) are considered as hidden layers.Therefore, layer-attributes consists of (L+1)
    values. The methods used here to initialise the values of parameters are Random and Xavier Initialisations.'''

def random_initialization(layer_attributes):
    
    L=len(layer_attributes)-1
    W=[]
    B=[]
    np.random.seed(10)
    for i in range(1,L+1):
        weight_i = np.random.uniform(-1,1,(layer_attributes[i],layer_attributes[i-1]))
        bias_i=np.zeros((layer_attributes[i],1))
        W.append(weight_i)
        B.append(bias_i)
        
    return W,B

def xavier_initialization(layer_attributes):
    
    L=len(layer_attributes)-1
    W=[]
    B=[]
    for i in range(1,L+1):
        lim = np.sqrt(6/(i+(i-1)))
        weight_i = np.random.uniform(-lim,lim,(layer_attributes[i],layer_attributes[i-1]))
        bias_i=np.zeros((layer_attributes[i],1))
        W.append(weight_i)
        B.append(bias_i)
        
    return W,B
        


# In[90]:


#FORWARD PROPAGATION
def forward_propagation(x,w,b,activation='sigmoid_function'):
    
    '''Forward propagation is used to find the predicted value of output and cost function by going forward,starting from 
    input layer until the output layer.We calculate the pre-activation and activation values and returns the latter after each
    layer. The input parameters taken are input data set,weights and bias value, and activation function to be used where the 
    default is set as sigmoid function. Softmax function is used to find the values at the output layer.
    Here,z is the linear part and a is the non-linear part(activation function) of a neuron.'''
    A=[]
    Z=[]
    length=len(w)
    #Hidden layers
    A.append(x)
    for i in range(length-1):
        z_i=np.dot(w[i],A[-1])+b[i]
        Z.append(z_i)
        if activation =='sigmoid_function':
            a_i = sigmoid_function(z_i)
            A.append(a_i)
        elif activation=='tanh_function':
            a_i = tanh_function(z_i)
            A.append(a_i)
        elif activation == 'relu_function':
            a_i = relu_function(z_i)
            A.append(a_i)
        elif activation == 'identity_function':
            a_i = identity_function(z_i)
            A.append(a_i)
    #output layer
    z_l = np.dot(w[-1],A[-1]) + b[-1]
    a_l = softmax_function(z_l)
    A.append(a_l)
    Z.append(z_l)

    return Z,A


# In[91]:


#BACK PROPAGATION
def back_propagation(A,y,W,B,Z,lambd,activation='sigmoid_function',loss='cross_entropy_function'):
    
    '''Back propagation is used to find the derivatives of each weights and biases at each layers by starting 
    from the output layer and travelling backwards.We find the derivatives wrto ouput layer,wrto hidden layer and eventually
    wrto weights and biases;dw=dJ/dw,db=dJ/db,dz=dJ/dz.'''
    m=y.shape[1]
    L=len(W)
    dW=[]
    dB=[]
    dZ=[]
    #Output Layer
    if loss=='cross_entropy_function':
        dZ.append(A[-1]-y)
        dB.append((1/m)*np.sum(dZ[-1],axis=1,keepdims=True))
        dW.append((1/m)*(np.dot(dZ[-1],A[-2].T))+(lambd/m)*W[-1])
    elif loss=='mean_squared_error_function':
        dZ.append((A[-1]-y)*A[-1]*(1-A[-1]))
        dB.append((1/m)*np.sum(dZ[-1],axis=1,keepdims=True))
        dW.append((1/m)*(np.dot(dZ[-1],A[-2].T))+(lambd/m)*W[-1])
        
   
    #Hidden layers
    l=L-1
    while l >0:
        if activation=='sigmoid_function':
            dz_l = (1/m)*np.dot(W[l].T,dZ[-1])*sigmoid_function_dash(A[l])
            db_l = (1/m)*np.sum(dz_l,axis=1,keepdims=True)
            dw_l = (1/m)*np.dot(dz_l,A[l-1].T) + (lambd/m)*W[l-1]
            dW.append(dw_l)
            dB.append(db_l)
            dZ.append(dz_l)
        
        elif activation == 'relu_function':
            dz_l = (1/m)*np.dot(W[l].T,dZ[-1])*relu_function_dash(A[l])
            db_l = (1/m)*np.sum(dz_l,axis=1,keepdims=True)
            dw_l = (1/m)*np.dot(dz_l,A[l-1].T) + (lambd/m)*W[l-1]
            dW.append(dw_l)
            dB.append(db_l)
            dZ.append(dz_l)
            
        elif activation=='tanh_function':
            dz_l = (1/m)*np.dot(W[l].T,dZ[-1])*tanh_function_dash(A[l])
            db_l = (1/m)*np.sum(dz_l,axis=1,keepdims=True)
            dw_l = (1/m)*np.dot(dz_l,A[l-1].T) + (lambd/m)*W[l-1]
            dW.append(dw_l)
            dB.append(db_l)
            dZ.append(dz_l)
            
        elif activation=='identity_function':
            dz_l = (1/m)*np.dot(W[l].T,dZ[-1])*identity_function_dash(A[l])
            db_l = (1/m)*np.sum(dz_l,axis=1,keepdims=True)
            dw_l = (1/m)*np.dot(dz_l,A[l-1].T) + (lambd/m)*W[l-1]
            dW.append(dw_l)
            dB.append(db_l)
            dZ.append(dz_l)
            
        l=l-1
            
    return dZ[::-1],dW[::-1],dB[::-1]


# In[92]:


#OPTIMIZERS
def gradient_descent(W,B,dW,dB,learning_rate):
    
    '''Mini batch,vanilla and stochastic gradient descents can be performed for this basic 
    gradient descent variant.'''
    
    alpha=learning_rate
    length=len(W)
    for i in range(length):
        W[i] = W[i] - alpha*dW[i]
        B[i] = B[i] - alpha*dB[i]
    return W,B

def momentum_gradient_descent(w,b,dw,db,learning_rate,momentum,update_w,update_b):
    
    '''Nesterov accelarated gradient descent can also be implemented as a special case of this Momentum gradient
    descent  where we find wlookahead before calculating gradients '''
    
    for i in range(len(w)):  
        update_w[i] = (momentum*update_w[i]) + dw[i]
        update_b[i] = (momentum*update_b[i]) + db[i]
        w[i] = w[i] - (learning_rate*update_w[i])
        b[i] = b[i] - (learning_rate*update_b[i])
    return w,b,update_w,update_b

def rms_prop(w,b,dw,db,learning_rate,beta,epsilon,v_t_w,v_t_b):
    
    '''RMSProp is an adaptive learning algorithm and the hyperparamteters used 
    here are learning_rate(alpha) and beta.'''
    
    for i in range(len(w)):
        v_t_w[i] = beta*v_t_w[i] + (1-beta)*((dw[i])**2)
        v_t_b[i] = beta*v_t_b[i] + (1-beta)*((db[i])**2)
        w[i] = w[i] - (learning_rate/(np.sqrt(v_t_w[i]+epsilon)))*dw[i]
        b[i] = b[i] - (learning_rate/(np.sqrt(v_t_b[i]+epsilon)))*db[i]
        return w,b,v_t_w,v_t_b


def adam(w,b,dw,db,learning_rate,beta1,beta2,epsilon,m_w,m_b,v_w,v_b,t):
    
    '''Adam(Adaptive moments) is one of the widely used adaptive learning gradient descent variant and it takes 
    beta1,beta2,learning_rate(alpha) as hyperparameters and also a small value epsilon to avoid zero division
    errors.'''
    
    m_w_cap=[]
    m_b_cap=[]
    v_w_cap=[]
    v_b_cap=[]
    for k in range(len(w)):
        upd_w=np.zeros((w[k].shape))
        upd_b=np.zeros((b[k].shape))
        m_w_cap.append(upd_w)
        m_b_cap.append(upd_b)
        v_w_cap.append(upd_w)
        v_b_cap.append(upd_b)
        
    for i in range(len(w)):
        m_w[i]=beta1*m_w[i]+(1-beta1)*dw[i]
        m_b[i]=beta1*m_b[i]+(1-beta1)*db[i]
        v_w[i]=beta2*v_w[i]+(1-beta2)*(dw[i]**2)
        v_b[i]=beta2*v_b[i]+(1-beta2)*(db[i]**2)
        m_w_cap[i] = (1/(1-math.pow(beta1,t)))*m_w[i]
        m_b_cap[i] = (1/(1-math.pow(beta1,t)))*m_b[i]
        v_w_cap[i] = (1/(1-math.pow(beta2,t)))*v_w[i]
        v_b_cap[i] = (1/(1-math.pow(beta2,t)))*v_b[i]
        w[i] = w[i] - (learning_rate/(np.sqrt(v_w_cap[i]+epsilon)))*m_w_cap[i]
        b[i] = b[i] - (learning_rate/(np.sqrt(v_b_cap[i]+epsilon)))*m_b_cap[i]
        
    return w,b,m_w,m_b,v_w,v_b

def nadam(w,b,dw,db,learning_rate,beta1,beta2,epsilon,m_w,m_b,v_w,v_b,t):
    
    '''NAdam(Nesterov Adam) is the nesterov gradient variant of Adam'''
    
    m_w_cap=[]
    m_b_cap=[]
    v_w_cap=[]
    v_b_cap=[]
    for k in range(len(w)):
        upd_w=np.zeros((w[k].shape))
        upd_b=np.zeros((b[k].shape))
        m_w_cap.append(upd_w)
        m_b_cap.append(upd_b)
        v_w_cap.append(upd_w)
        v_b_cap.append(upd_b)
        
    for i in range(len(w)):
        
        m_w[i] = beta1*m_w[i] + (1-beta1)*dw[i]
        m_b[i] = beta1*m_b[i] + (1-beta1)*db[i]
        v_w[i] = beta2*v_w[i] + (1-beta2)*(dw[i])**2
        v_b[i] = beta2*v_b[i] + (1-beta2)*(db[i])**2
        m_w_cap[i] = (1/(1-beta1**t))*m_w[i]
        m_b_cap[i] = (1/(1-beta1**t))*m_b[i]
        v_w_cap[i] = (1/(1-beta2**t))*v_w[i]
        v_b_cap[i] = (1/(1-beta2**t))*v_b[i]
        w[i] = w[i]-(learning_rate/np.sqrt(v_w_cap[i]+epsilon))*(beta1*m_w_cap[i]+(1-beta1)*dw[i]/(1-beta1**t))
        b[i] = b[i]-(learning_rate/np.sqrt(v_b_cap[i]+epsilon))*(beta1*m_b_cap[i]+(1-beta1)*db[i]/(1-beta1**t))
        
    return w,b,m_w,m_b,v_w,v_b
        
                                


# In[93]:


#ACCURACY
def accuracy(y,yout):
    '''Function to find the accuracy taking y and ypred as input and returns accracy value.'''
    yout=np.argmax(yout,axis=0)
    y = np.argmax(y,axis=0)   
    acc=np.mean(y==yout)*100
    return acc
    


# In[94]:


#FUNCTION FOR PLOTS
def plot_error(j_train, j_val):
    plt.plot(list(range(len(j_train))), j_train, 'r', label="Train Loss")
    plt.plot(list(range(len(j_val))), j_val, 'lime', label="Validation Loss")
    plt.title("Training and Validation Loss vs No. of Epochs", size=16)
    plt.xlabel("No. of epochs", size=16)
    plt.ylabel("Loss", size=16)
    plt.legend()
    plt.grid()
    plt.show()
    
def plot_accuracy(acc_train, acc_val):
    plt.plot(list(range(len(acc_train))), acc_train, 'r', label="Train Accuracy")
    plt.plot(list(range(len(acc_val))), acc_val, 'lime', label="Validation Accuracy")
    plt.title("Training and Validation Accuracy vs No. of Epochs", size=16)
    plt.xlabel("No. of epochs", size=16)
    plt.ylabel("Accuracy", size=16)
    plt.legend()
    plt.grid()
    plt.show()


# In[97]:


 #PREDICT FUNCTION
def predict(x,y,w,b,lambd,activation='relu_function'):
    '''This function is to predict the cost and accuracy values of the test data
       input :  x(input)
                y(output)
                w,b(weights and biases)
                lambd(regularization parameter)
                loss(loss function)
                activation(activation function)'''
    
    z,a = forward_propagation(x,w,b,activation)
    cost_test = mean_squared_error(y,a[-1],w,lambd)
    acc= accuracy(y,a[-1])
            
    return acc,cost_test


# In[98]:


def neural_network(x_train,y_train,x_val,y_val,learning_rate = 0.001,momentum = 0.9,beta=0.9,beta1=0.9,beta2=0.99,epochs = 20,num_hidden_layers = 1,neurons=64,batch_size=16,epsilon=0.00000001,weight_init='random_initialization',
                  activation='relu_function',loss='cross_entropy_function',optimizer='adam',lambd=0,wandb_project="CS6910_DL_ASS1"):
 
    wandb.init(project=wandb_project)
    run_name= f'lr_{learning_rate}_acti_{activation}_w_in{weight_init}_opt_{optimizer}_epoch_{epochs}_num_hid_{num_hidden_layers}_loss_{loss}_batchsize_{batch_size}_neur_{neurons}_lam_{lambd}_momentum_{momentum}_beta_{beta}_beta1_{beta1}_beta2_{beta2}'
    print(run_name)
    
    '''initializing required attributes'''
    layer=[]
    n,m=x_train.shape
    J_train=[]
    Accuracy_train=[]
    J_val=[]
    Accuracy_val=[]
    acc_train=0
    acc_val=0
    layer.append(x_train.shape[0])
    for i in range(num_hidden_layers):
        layer.append(neurons)
    layer.append(y_train.shape[0])
    print(f'neuron configuration: {layer}')
    if weight_init=='random_initialization':
        w,b=random_initialization(layer)
    elif weight_init=='xavier_initialization':
        w,b=xavier_initialization(layer)
    update_w=[]
    update_b=[]
    w_lookahead=[]
    b_lookahead=[]
    v_t_w=[]
    v_t_b=[]
    m_w=[]
    m_b=[]
    v_w=[]
    v_b=[]
    for k in range(len(w)):
        upd_w=np.zeros((w[k].shape))
        upd_b=np.zeros((b[k].shape))
        update_w.append(upd_w)
        w_lookahead.append(upd_w)
        update_b.append(upd_b)
        b_lookahead.append(upd_b)
        v_t_w.append(upd_w)
        v_t_b.append(upd_b)
        m_w.append(upd_w)
        m_b.append(upd_b)
        v_w.append(upd_w)
        v_b.append(upd_b)
        
    num_batches = x_train.shape[1]//batch_size
    
    if optimizer=='vanilla_gradient_descent':
        
        for j in range(epochs):
            z,a = forward_propagation(x_train,w,b,activation)
            dz,dw,db=back_propagation(a,y_train,w,b,z,lambd,activation,loss)
            w,b=gradient_descent(w,b,dw,db,learning_rate)
            if loss=='cross_entropy_function':
                
                cost_train=cross_entropy_function(y_train,a[-1],w,lambd)
                J_train.append(cost_train)
                acc_train = accuracy(y_train,a[-1])
                Accuracy_train.append(acc_train)
                z_val,a_val = forward_propagation(x_val,w,b,activation)
                cost_val = cross_entropy_function(y_val,a_val[-1],w,lambd)
                J_val.append(cost_val)
                acc_val=accuracy(y_val,a_val[-1])
                Accuracy_val.append(acc_val)
            elif loss=='mean_squared_error_function':
                
                cost_train=mean_squared_error_function(y_train,a[-1],w,lambd)
                J_train.append(cost_train)
                acc_train = accuracy(y_train,a[-1])
                Accuracy_train.append(acc_train)
                z_val,a_val = forward_propagation(x_val,w,b,activation)
                cost_val = mean_squared_error_function(y_val,a_val[-1],w,lambd)
                J_val.append(cost_val)
                acc_val = accuracy(y_val,a_val[-1])
                Accuracy_val.append(acc_val)
            wandb.log({"accuracy_train": acc_train, "accuracy_validation": acc_val, "loss_train": cost_train, "cost_validation": cost_val, 'epochs': j})
            
            if j%(epochs/10)==0:
                print(f' \n epoch:{j:4d}  Train error:  {J_train[-1]:8.2f}  Train accuracy: {Accuracy_train[-1]:8.2f} Val error: {J_val[-1]:8.2f} Val accuracy: {Accuracy_val[-1]:8.2f}')
            
    if optimizer=='stochastic_gradient_descent':
        
         for j in range(epochs):
            for i in range(num_batches):
                x_mb = x_train[:,i*batch_size:(i+1)*batch_size].reshape(x_train.shape[0],batch_size)
                y_mb = y_train[:,i*batch_size:(i+1)*batch_size].reshape(y_train.shape[0],batch_size)
                z,a = forward_propagation(x_mb,w,b,activation)
                dz,dw,db=back_propagation(a,y_mb,w,b,z,lambd,activation,loss)
                w,b=gradient_descent(w,b,dw,db,learning_rate)
            if x_train.shape[1] % batch_size !=0:
                x_last = x_train[:,num_batches*batch_size:]
                y_last = y_train[:,num_batches*batch_size:]
                dz,dw,db=back_propagation(a,y_last,w,b,z,lambd,activation,loss)
                w,b=gradient_descent(w,b,dw,db,learning_rate)
            z,a = forward_propagation(x_train,w,b,activation)
            cost_train=cross_entropy_function(y_train,a[-1],w,lambd)
            J_train.append(cost_train)
            acc_train = accuracy(y_train,a[-1])
            Accuracy_train.append(acc_train)
            z_val,a_val = forward_propagation(x_val,w,b,activation)
            cost_val = cross_entropy_function(y_val,a_val[-1],w,lambd)
            J_val.append(cost_val)
            acc_val = accuracy(y_val,a_val[-1])
            wandb.log({"accuracy_train": acc_train, "accuracy_validation": acc_val, "loss_train": cost_train, "cost_validation": cost_val, 'epochs': j})
            Accuracy_val.append(acc_val)
            if j%(epochs/10)==0:
                print(f' \n epoch:{j:4d}  Train error:  {J_train[-1]:8.2f}  Train accuracy: {Accuracy_train[-1]:8.2f} Val error: {J_val[-1]:8.2f} Val accuracy: {Accuracy_val[-1]:8.2f}')
            
    if optimizer=='momentum_gradient_descent':
            
        for j in range(epochs):
            
            for i in range(num_batches):
                x_mb = x_train[:,i*batch_size:(i+1)*batch_size].reshape(x_train.shape[0],batch_size)
                y_mb = y_train[:,i*batch_size:(i+1)*batch_size].reshape(y_train.shape[0],batch_size)
                z,a = forward_propagation(x_mb,w,b,activation)
                dz,dw,db=back_propagation(a,y_mb,w,b,z,lambd,activation,loss)
                w,b,update_w,update_b = momentum_gradient_descent(w,b,dw,db,learning_rate,momentum,update_w,update_b)
            if x_train.shape[1] % batch_size !=0:
                x_last = x_train[:,num_batches*batch_size:]
                y_last = y_train[:,num_batches*batch_size:]
                z,a = forward_propagation(x_last,w,b,activation)
                dz,dw,db=back_propagation(a,y_last,w,b,z,lambd,activation,loss)
                w,b,update_w,update_b = momentum_gradient_descent(w,b,dw,db,learning_rate,momentum,update_w,update_b)
            z,a = forward_propagation(x_train,w,b,activation)
            if loss=='cross_entropy_function':
                
                cost_train=cross_entropy_function(y_train,a[-1],w,lambd)
                J_train.append(cost_train)
                acc_train = accuracy(y_train,a[-1])
                Accuracy_train.append(acc_train)
                z_val,a_val = forward_propagation(x_val,w,b,activation)
                cost_val = cross_entropy_function(y_val,a_val[-1],w,lambd)
                J_val.append(cost_val)
                acc_val=accuracy(y_val,a_val[-1])
                Accuracy_val.append(acc_val)
            elif loss=='mean_squared_error_function':
                
                cost_train=mean_squared_error_function(y_train,a[-1],w,lambd)
                J_train.append(cost_train)
                acc_train = accuracy(y_train,a[-1])
                Accuracy_train.append(acc_train)
                z_val,a_val = forward_propagation(x_val,w,b,activation)
                cost_val = mean_squared_error_function(y_val,a_val[-1],w,lambd)
                J_val.append(cost_val)
                acc_val = accuracy(y_val,a_val[-1])
                Accuracy_val.append(acc_val)
            wandb.log({"accuracy_train": acc_train, "accuracy_validation": acc_val, "loss_train": cost_train, "cost_validation": cost_val, 'epochs': j})
            
            if j%(epochs/10)==0:
                print(f' \n epoch:{j:4d}  Train error:  {J_train[-1]:8.2f}  Train accuracy: {Accuracy_train[-1]:8.2f} Val error: {J_val[-1]:8.2f} Val accuracy: {Accuracy_val[-1]:8.2f}')
                
                
    if optimizer == 'nesterov_accelarated_gradient_descent':
        
        for j in range(epochs):
            
            for i in range(num_batches):
                x_mb = x_train[:,i*batch_size:(i+1)*batch_size].reshape(x_train.shape[0],batch_size)
                y_mb = y_train[:,i*batch_size:(i+1)*batch_size].reshape(y_train.shape[0],batch_size)
                for i in range(len(w)):
                    w_lookahead[i] = w[i]-momentum*update_w[i]
                    b_lookahead[i] = b[i]-momentum*update_b[i]
                z,a = forward_propagation(x_mb,w_lookahead,b_lookahead,activation)
                dz,dw,db=back_propagation(a,y_mb,w_lookahead,b_lookahead,z,lambd,activation,loss)
                w,b,update_w,update_b = momentum_gradient_descent(w,b,dw,db,learning_rate,momentum,update_w,update_b)
            if x_train.shape[1] % batch_size !=0:
                x_last = x_train[:,num_batches*batch_size:]
                y_last = y_train[:,num_batches*batch_size:]
                z,a = forward_propagation(x_last,w_lookahead,b_lookahead,activation)
                dz,dw,db=back_propagation(a,y_last,w_lookahead,b_lookahead,z,lambd,activation,loss)
                w,b,update_w,update_b = momentum_gradient_descent(w,b,dw,db,learning_rate,momentum,update_w,update_b)
                
            z,a = forward_propagation(x_train,w,b,activation)
            if loss=='cross_entropy_function':
                
                cost_train=cross_entropy_function(y_train,a[-1],w,lambd)
                J_train.append(cost_train)
                acc_train = accuracy(y_train,a[-1])
                Accuracy_train.append(acc_train)
                z_val,a_val = forward_propagation(x_val,w,b,activation)
                cost_val = cross_entropy_function(y_val,a_val[-1],w,lambd)
                J_val.append(cost_val)
                acc_val=accuracy(y_val,a_val[-1])
                Accuracy_val.append(acc_val)
            elif loss=='mean_squared_error_function':
                
                cost_train=mean_squared_error_function(y_train,a[-1],w,lambd)
                J_train.append(cost_train)
                acc_train = accuracy(y_train,a[-1])
                Accuracy_train.append(acc_train)
                z_val,a_val = forward_propagation(x_val,w,b,activation)
                cost_val = mean_squared_error_function(y_val,a_val[-1],w,lambd)
                J_val.append(cost_val)
                acc_val = accuracy(y_val,a_val[-1])
                Accuracy_val.append(acc_val)
            wandb.log({"accuracy_train": acc_train, "accuracy_validation": acc_val, "loss_train": cost_train, "cost_validation": cost_val, 'epochs': j})
            
            if j%(epochs/10)==0:
                print(f' \n epoch:{j:4d}  Train error:  {J_train[-1]:8.2f}  Train accuracy: {Accuracy_train[-1]:8.2f} Val error: {J_val[-1]:8.2f} Val accuracy: {Accuracy_val[-1]:8.2f}')
                
                
    if optimizer == 'rms_prop':
            
        for j in range(epochs):
            
            for i in range(num_batches):
                x_mb = x_train[:,i*batch_size:(i+1)*batch_size].reshape(x_train.shape[0],batch_size)
                y_mb = y_train[:,i*batch_size:(i+1)*batch_size].reshape(y_train.shape[0],batch_size)
                z,a = forward_propagation(x_mb,w,b,activation)
                dz,dw,db=back_propagation(a,y_mb,w,b,z,lambd,activation,loss)
                w,b,v_t_w,v_t_b = rms_prop(w,b,dw,db,learning_rate,beta,epsilon,v_t_w,v_t_b)
            if x_train.shape[1] % batch_size !=0:
                x_last = x_train[:,num_batches*batch_size:]
                y_last = y_train[:,num_batches*batch_size:]
                z,a = forward_propagation(x_last,w,b,activation)
                dz,dw,db=back_propagation(a,y_last,w,b,z,lambd,activation,loss)
                w,b,v_t_w,v_t_b = rms_prop(w,b,dw,db,learning_rate,beta,epsilon,v_t_w,v_t_b)
            z,a = forward_propagation(x_train,w,b,activation)
            if loss=='cross_entropy_function':
                
                cost_train=cross_entropy_function(y_train,a[-1],w,lambd)
                J_train.append(cost_train)
                acc_train = accuracy(y_train,a[-1])
                Accuracy_train.append(acc_train)
                z_val,a_val = forward_propagation(x_val,w,b,activation)
                cost_val = cross_entropy_function(y_val,a_val[-1],w,lambd)
                J_val.append(cost_val)
                acc_val=accuracy(y_val,a_val[-1])
                Accuracy_val.append(acc_val)
            elif loss=='mean_squared_error_function':
                
                cost_train=mean_squared_error_function(y_train,a[-1],w,lambd)
                J_train.append(cost_train)
                acc_train = accuracy(y_train,a[-1])
                Accuracy_train.append(acc_train)
                z_val,a_val = forward_propagation(x_val,w,b,activation)
                cost_val = mean_squared_error_function(y_val,a_val[-1],w,lambd)
                J_val.append(cost_val)
                acc_val = accuracy(y_val,a_val[-1])
                Accuracy_val.append(acc_val)
            wandb.log({"accuracy_train": acc_train, "accuracy_validation": acc_val, "loss_train": cost_train, "cost_validation": cost_val, 'epochs': j})
            
            if j%(epochs/10)==0:
                print(f' \n epoch:{j:4d}  Train error:  {J_train[-1]:8.2f}  Train accuracy: {Accuracy_train[-1]:8.2f} Val error: {J_val[-1]:8.2f} Val accuracy: {Accuracy_val[-1]:8.2f}')
                        
                
    if optimizer == 'adam':
        for  j in range(epochs):
            
            for i in range(num_batches):
                x_mb = x_train[:,i*batch_size:(i+1)*batch_size].reshape(x_train.shape[0],batch_size)
                y_mb = y_train[:,i*batch_size:(i+1)*batch_size].reshape(y_train.shape[0],batch_size)
                z,a = forward_propagation(x_mb,w,b,activation)
                dz,dw,db=back_propagation(a,y_mb,w,b,z,lambd,activation,loss)
                w,b,m_w,m_b,v_w,v_b = adam(w,b,dw,db,learning_rate,beta1,beta2,epsilon,m_w,m_b,v_w,v_b,j+1)
            if x_train.shape[1] % batch_size !=0:
                x_last = x_train[:,num_batches*batch_size:]
                y_last = y_train[:,num_batches*batch_size:]
                z,a = forward_propagation(x_last,w,b,activation)
                dz,dw,db=back_propagation(a,y_last,w,b,z,lambd,activation,loss)
                w,b,m_w,m_b,v_w,v_b = adam(w,b,dw,db,learning_rate,beta1,beta2,epsilon,m_w,m_b,v_w,v_b,j+1)
            z,a=forward_propagation(x_train,w,b,activation)
            if loss=='cross_entropy_function':
                
                cost_train=cross_entropy_function(y_train,a[-1],w,lambd)
                J_train.append(cost_train)
                acc_train = accuracy(y_train,a[-1])
                Accuracy_train.append(acc_train)
                z_val,a_val = forward_propagation(x_val,w,b,activation)
                cost_val = cross_entropy_function(y_val,a_val[-1],w,lambd)
                J_val.append(cost_val)
                acc_val=accuracy(y_val,a_val[-1])
                Accuracy_val.append(acc_val)
            elif loss=='mean_squared_error_function':
                
                cost_train=mean_squared_error_function(y_train,a[-1],w,lambd)
                J_train.append(cost_train)
                acc_train = accuracy(y_train,a[-1])
                Accuracy_train.append(acc_train)
                z_val,a_val = forward_propagation(x_val,w,b,activation)
                cost_val = mean_squared_error_function(y_val,a_val[-1],w,lambd)
                J_val.append(cost_val)
                acc_val = accuracy(y_val,a_val[-1])
                Accuracy_val.append(acc_val)
            wandb.log({"accuracy_train": acc_train, "accuracy_validation": acc_val, "loss_train": cost_train, "cost_validation": cost_val, 'epochs': j})
            
            if j%(epochs/10)==0:
                print(f' \n epoch:{j:4d}  Train error:  {J_train[-1]:8.2f}  Train accuracy: {Accuracy_train[-1]:8.2f} Val error: {J_val[-1]:8.2f} Val accuracy: {Accuracy_val[-1]:8.2f}')
                
                
    if optimizer =='nadam':
        for  j in range(epochs):
            
            for i in range(num_batches):
                x_mb = x_train[:,i*batch_size:(i+1)*batch_size].reshape(x_train.shape[0],batch_size)
                y_mb = y_train[:,i*batch_size:(i+1)*batch_size].reshape(y_train.shape[0],batch_size)
                z,a = forward_propagation(x_mb,w,b,activation)
                dz,dw,db=back_propagation(a,y_mb,w,b,z,lambd,activation,loss)
                w,b,m_w,m_b,v_w,v_b = nadam(w,b,dw,db,learning_rate,beta1,beta2,epsilon,m_w,m_b,v_w,v_b,j+1)
            if x_train.shape[1] % batch_size !=0:
                x_last = x_train[:,num_batches*batch_size:]
                y_last = y_train[:,num_batches*batch_size:]
                z,a = forward_propagation(x_last,w,b,activation)
                dz,dw,db=back_propagation(a,y_last,w,b,z,lambd,activation,loss)
                w,b,m_w,m_b,v_w,v_b = nadam(w,b,dw,db,learning_rate,beta1,beta2,epsilon,m_w,m_b,v_w,v_b,j+1)
            z,a=forward_propagation(x_train,w,b,activation)
            
            if loss=='cross_entropy_function':
                
                cost_train=cross_entropy_function(y_train,a[-1],w,lambd)
                J_train.append(cost_train)
                acc_train = accuracy(y_train,a[-1])
                Accuracy_train.append(acc_train)
                z_val,a_val = forward_propagation(x_val,w,b,activation)
                cost_val = cross_entropy_function(y_val,a_val[-1],w,lambd)
                J_val.append(cost_val)
                acc_val=accuracy(y_val,a_val[-1])
                Accuracy_val.append(acc_val)
            elif loss=='mean_squared_error_function':
                
                cost_train=mean_squared_error_function(y_train,a[-1],w,lambd)
                J_train.append(cost_train)
                acc_train = accuracy(y_train,a[-1])
                Accuracy_train.append(acc_train)
                z_val,a_val = forward_propagation(x_val,w,b,activation)
                cost_val = mean_squared_error_function(y_val,a_val[-1],w,lambd)
                J_val.append(cost_val)
                acc_val = accuracy(y_val,a_val[-1])
                Accuracy_val.append(acc_val)
            wandb.log({"accuracy_train": acc_train, "accuracy_validation": acc_val, "loss_train": cost_train, "cost_validation": cost_val, 'epochs': j})
            
            if j%(epochs/10)==0:
                print(f' \n epoch:{j:4d}  Train error:  {J_train[-1]:8.2f}  Train accuracy: {Accuracy_train[-1]:8.2f} Val error: {J_val[-1]:8.2f} Val accuracy: {Accuracy_val[-1]:8.2f}')
        
            
            
    plot_error(J_train,J_val)
    plot_accuracy(Accuracy_train,Accuracy_val)
    wandb.run.name = run_name
    return w,b


# In[ ]:


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb_project',default="CS6910_DL_ASS1")
    parser.add_argument('--data_set', help='FASHION-MNIST OR MNIST',default='fashion_mnist')
    parser.add_argument('--learning_rate', type=float, default =0.001,
                    help='initial learning rate for gradient descent')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--num_hidden_layers', type=int, default=1, help='number of hidden layers')
    parser.add_argument('--neurons',type=int, default=64, help='number of neurons in hidden layers')
    parser.add_argument('--activation', default='relu_function',help='activation function (tanh_function or sigmoid_function or relu_function or identity_function)')
    parser.add_argument('--loss', default='cross_entropy_function',
                    help='loss function (cross_entropy_function or mean_squared_error_function)')
    parser.add_argument('--optimizer', default='adam',
      help='optimizers (stochastic_gradient_descent or momentum_gradient_descent or nesterov_accelarated_gradient_descent or adam or nadam or rms_prop)')
    parser.add_argument('--batch_size',type=int,default=8)
    parser.add_argument('--beta', type=float, default=0.9)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.99)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--weight_init', default='random_initialization',
                    help='weight initialization(random_initialization or xavier_initialization)')
    parser.add_argument('--lambd', type=float, default=0)
    parser.add_argument('--epsilon', type=float, default=1e-8)

    

    args = parser.parse_args()
    if args.data_set=="fashion_mnist":
        x_train,y_train,x_val,y_val,x_test,y_test=pre_processing_data(X_train,Y_train,X_test,Y_test)
        x_train=data_augmentation(x_train)
    elif args.data_set=="mnist":
        x_train,y_train,x_val,y_val,x_test,y_test=pre_processing_data(X_train_mnist,Y_train_mnist,X_test_mnist,Y_test_mnist)
        x_train=data_augmentation(x_train)
    w,b=neural_network(x_train,y_train,x_val,y_val,args.learning_rate,args.momentum,args.beta,args.beta1,args.beta2,args.epochs,args.num_hidden_layers,args.neurons,args.batch_size,args.epsilon,args.weight_init,
                  args.activation,args.loss,args.optimizer,args.lambd,args.wandb_project)
    
    wandb.run.save()
    wandb.run.finish()
    


# In[ ]:




