  *feedforwardneuralnet*  -This file includes the complete code of the project.   
  *train.py* -This file can be used for running the code in the terminal.  

The following packages are used for completing the assignment:-  
  **Numpy** - matrix operations.  
  **WANDB** - performing sweep and analysis results from plots, confusion matrix etc.  
  **scikit-learn** - plotting confusion matrix in the jupyter notebook, test-train-split of dataset.  
  **keras** - importing FASHION-MNIST and MNIST datasets.  
  **seaborn** - To plot the confusion Matrix
  

Further details about the implementation is given below:-  

### FUNCTIONS USED:-  

1) **pre_processing_data** : This function in train.py performs the following tasks:-  
     Changes the shape of input data and ground truth.  
     Converts the output data into one hot vector form.  
     Creates a validation dataset.  
     Normalizes data.  
     
2) **data_augmentation** : This function in train.py adds a gaussian noise to the train data.  
   mean of gaussian random noise:1  
   variance :0.01  
   
3) **Activation functions and its derivatives** :

   a) sigmoid_function = 1/((1+exp(-z))
   sigmoid_function_dash : This function is the derivative of sigmoid_function = sigmoid_function(z)x(1-sigmoid_function(z))  
                                        
   b) relu_function(z) = max(0,z)  
   Its derivative is given as relu_function_dash(z)    
   
   c) identity_function(z) = z  
   Its derivative is given by identity_function_dash(z)    
   
   d) tanh_function(z) : This denotes the tanh activation function  
   Its derivative is given by tanh_function_dash(z)
   
   e) softmax_function(z) : Since the given problem is a multi-class classification(with 10 labels), we use softmax function for the output layer.  
    Its derivative is given by softmax_function_dash(z)  
    
    Mostly, these activation functions are used in forward propagation and its derivative functions are used in back propagation.  
    
4) **Loss functions** :  

   a) cross_entropy_function : For multi-label problems, cross entropy function is the recommended loss function, which gives a higher penalty in case of misclassifications
   and therefore, minimizes better. 
      
   b) mean_squared_error_function : Here, Mean squared error loss is used to compare the results obtained with cross entropy loss.  
   
   
5) **Weight initialization** :  

   a) random_initialization : Here , weights are initialized randomly as a uniform distribution between limits -1 and 1 where biases are either initialized to zero or a
     positive value according to the sweeps.  
      
   b) xavier_initialization : Here, weights are initialized as a uniform distribution between the specified limits and bias to zero.  
   
   
6) **forward_propagation** : Forward propagation is used to find the predicted value of output and cost function by going forward,starting from input layer until the output layer.We calculate the pre-activation and activation values and returns the latter after each layer. The input parameters taken are input data set,weights and bias value,and activation function to be used where the default is set as sigmoid function. Softmax function is used to find the values at the output layer. Here,z is the linear part and is the non-linear part(activation function) of a neuron.   
     z_l = w_l * a_l-1 + b_l, where z_l is the pre-activation part  
     a_l = g(z_l), where a_l is the activation part  
     
 7) **back_propagation** : Back propagation is used to find the derivatives of each weights and biases at each layers by starting from the output layer and travelling backwards.We find the derivatives wrto ouput layer,wrto hidden layer and eventuallywrto weights and biases.  
 dw=dJ/dw,db=dJ/db,dz=dJ/dz  
 
 8) **Gradient_descent_variations** : 
 
    a) gradient_descent : This function is used to implement vanilla gradient descent and stochastic gradient descent  
    b) momentum_gradient_descent : This function written is used to Momentum gradient descent and Nesterov accelarated gradient descent  
    c) adam: This function is used to implement Adam(Adaptive moments)  
    d) nadam : This function is used to implement Nesterov Adam  
    e) rms_prop : This function is used to implement RMS-Prop Algorithm  
    
    
 9) **accuracy** : Function to find accuracy of given data    

 10) **plot_error** : Function used to plot train and validation error  

 11) **plot_accuracy** : Function used to plot train and validation accuracy  

 12) **predict** : Function used to predict output value of test data and find the test accuracy and test error in feed_forward_NN 

 13) **neural_network** : This is the main function where the main function call is performed and also the sweep.  
   
      **HYPERPARAMETERS** :   
                            a. learning_rate(Learning rate)    
                            b. lambd(Regularization parameter)    
                            c. activation(Activation function)   
                            d. loss(Loss function)    
                            e. epsilon(used in gradient_descent algorithms to avoid zero division error)  
                            f. momentum(Momentum used in Momentum gradient descent and NAG)  
                            g. beta(used in rms_prop)  
                            h. beta1(used in adam and nadam)  
                            i. beta2(used in adam and nadam)  
                            j. wandb_project(project_name in wandb)  
                            k. neurons(Number of neurons in hidden layers)  
                            l. num_hidden_layers(number of hidden layers)  
                            m. batch_size(step size in gradient descent)  
                            n. epochs(Number of epochs) 
                            o. optimizer(gradient descent algorithm used)  
                           
   
 ## OUTCOMES:-  
 
 All the class labels was found out and displayed.  
 Feed forward Neural Network was successfully implemented.  
 Backpropogation algorithm was implemented without using any automatic differentiation packages.  
 All the spcified gradient descent algorithms have been implemented.  
 A total of 491 sweeps was performed and the hyperparameter configuration of highest validation accuracy was found  
 The cross entropy and mean squared error cost functions was compared based on results.  
 The confusion matrix was plotted for the highest validation accuracy case.  
 Analysis was done on MNIST dataset as well with the inferences obtained from Fashion-MNIST dataset.  
 *train.py* file was created which could be run in the terminal and incudes all the specified hyperparameter choices. 
 
 
 ### HYPER-PARAMETERS FOR HIGHEST VALIDATION ACCURACY:-  
 
 learning_rate = 0.001  
 lambd = 0  
 activation = relu_function  
 loss = cross_entropy_function  
 epsilon = 1e-8  
 momentum = 0.9  
 beta = 0.9  
 beta1 = 0.9  
 beta2 = 0.99  
 optimizer = adam   
 neurons = 64  
 num_hidden_layers = 1  
 batch_size = 8  
 epochs = 20  
 
 **HIGHEST VALIDATION ACCURACY FOR FASHION-MNIST DATASET = 87.25%**   
 **HIGHEST VALIDATION ACCURACY FOR MNIST DATASET = 97.5%**   
