'''
Fully Connected Neural Network from Scratch using Numpy and Pandas
Author : Arijit Mukherjee
Date   : Nov 2017
Task   : Predict Poker Hand
'''

'''
Import Required System Libraries
'''
import numpy as np
import pandas as pd
import math,sys
import pickle

'''
This function is for a beautiful progressbar that is handy during training
'''
#https://gist.github.com/rougier/c0d31f5cbdaac27b876c
def progress(value,  length=40, title = " ", vmin=0.0, vmax=1.0):
    # Block progression is 1/8
    blocks = ["", "▏","▎","▍","▌","▋","▊","▉","█"]
    vmin = vmin or 0.0
    vmax = vmax or 1.0
    lsep, rsep = "▏", "▕"
    value = min(max(value, vmin), vmax)
    value = (value-vmin)/float(vmax-vmin)
    v = value*length
    x = math.floor(v) # integer part
    y = v - x         # fractional part
    base = 0.125      # 0.125 = 1/8
    prec = 3
    i = int(round(base*math.floor(float(y)/base),prec)/base)
    bar = "█"*x + blocks[i]
    n = length-len(bar)
    bar = lsep + bar + " "*n + rsep
    sys.stdout.write("\r" + title + bar + " %.1f%%" % (value*100))
    sys.stdout.flush()


'''
Original Model Starts Here
Inside neural_network class all the Required methods are implemented
'''
class neural_net():
    '''
    Init Neural Network using requred params
    lr          : Learning Rate
    mom         : Momentum alpha
    lamda       : Regularizer Lambda
    max_steps   : Number of Epochs
    ni          : Input shape
    nh          : Hidden Layer eg [15,10] for two hidden layers with 15 and 10 neurons
    no          : Number of Outputs
    '''
    def __init__(self,ni,nh,no,lr=0.01,mom=0.08,max_steps=10,lamda=0.1):
        self._lr=lr
        self._lambda=lamda
        self._max_steps=max_steps
        self._n_hidden=nh
        self._n_output=no
        self._n_input=ni
        self._layers=[]
        self._labels=[]
        self._momentum=mom
        last_len=self._n_input+1
        idx=1
        for e in self._n_hidden:
            hidden_layer = [{'weights':2*np.random.rand(last_len)-1,'last_update':0.0,'name':'hidden_layer_'+str(idx)+'_unit_'+str(i+1)} for i in range(e)]
            self._layers.append(hidden_layer)
            last_len=e+1
            idx+=1
        output_layer= [{'weights':2*np.random.rand(last_len)-1,'last_update':0.,'name':'output_layer_unit_'+str(i+1)} for i in range(self._n_output)]
        self._layers.append(output_layer)

    '''
    Embded Method one hot encodes both X and Y
    '''
    def embed(self,X,Y):
        tX=[]
        for col in X.columns.values:
            tX.append(pd.get_dummies(X[col]))
        X=pd.concat(tX,axis=1)
        X=X.as_matrix()
        if Y is not None:
            Y=Y.as_matrix()
            self._labels=list(set(Y.flatten()))
            labels=dict(enumerate(self._labels))
            labels={v: k for k, v in labels.items()}
            new_y=np.zeros((Y.shape[0],self._n_output))
            for i in range(Y.shape[0]):
                new_y[i][labels[Y[i][0]]]=1
            Y=new_y
        return X,Y

    '''
    This method calculates W^TX
    inputs  : X
    weights : W
    '''
    def output(self,inputs,weights):
        if inputs.shape[0]!=weights.shape[0]:
            return np.dot(weights[:-1].T,inputs)+weights[-1]
        return np.dot(weights.T,inputs)

    '''
    Given x this mehtod calculates sigmoid(x)
    '''
    def sigmoid_forward(self,x):
        return 1./(1.+np.exp(-x))

    '''
    Given sigmoid(x) this method calculates derivative of sigmoid
    return sigmoid(x)(1-sigmoid(x))
    '''
    def sigmoid_prime(self,x):
        return x*(1.0-x)


    '''
    Given an input vector this method forward propagates the vector in each
    layer upto the Output layer
    returns values of last layer in a list
    '''
    def forward_propagate(self,row):
        inputs=row
        for layer in self._layers:
            new_inputs=[]
            for unit in layer:
                x=self.output(inputs,unit['weights'])
                unit['output']=self.sigmoid_forward(x)
                new_inputs.append(unit['output'])
            inputs=np.array(new_inputs)
        return inputs


    '''
    Given an expected output vector this method calculates error in each layer
    starting from the output layer upto the 1st hidden layer and backpropagates
    error and calcualtes delta_w
    '''
    def backward_propagate(self,y):
        for i in reversed(range(len(self._layers))):
            layer=self._layers[i]
            if i==(len(self._layers)-1):
                for j in range(len(layer)):
                    layer[j]['delta']=(layer[j]['output']-y[j])
            else:
                for j in range(len(layer)):
                    error=0.
                    for unit in self._layers[i+1]:
                        error+=unit['weights'][j]*unit['delta']
                    layer[j]['delta']=error*self.sigmoid_prime(layer[j]['output'])

    '''
    Given an input vector this method updates the neural network weights on each
    layer with del_w*learning_rate with regularizer and momentum
    '''
    def update_params(self,row):
        for i in range(len(self._layers)):
            layer=self._layers[i]
            inputs=np.append([row],[1.])
            if(i!=0):
                inputs=[unit['output'] for unit in self._layers[i-1]]
                inputs.append(1.0)
                inputs=np.array(inputs)
            for unit in self._layers[i]:
                '''
                momentum is implemented here
                '''
                unit['last_update']=self._momentum*unit['last_update']-self._lr*(unit['delta']*(np.array(inputs))+self._lambda*(unit['weights']))
                unit['weights']+=unit['last_update']


    '''
    Given two matrix predicted Y and True Y this method calcualtes the cross
    entropy loss with the regularizer .
    '''
    def cross_entropy(self,y,output):
        loss=0
        m=y.shape[0]
        w=0
        for layer in self._layers:
            for unit in layer:
                w+=np.sum(unit['weights']**2)
        loss+=self._lambda*(w/-m)
        for i in range(y.shape[0]):
            for j in range(y.shape[1]):
                loss+=y[i][j]*np.log(output[i][j])+(1-y[i][j])*np.log(1-output[i][j])
        return loss/-m

    '''
    This method uses all the above methods first embedding the X and Ys then
    for EACH in EPOCHS
        for EACH <x,y> in Dataset
            FORWARD_PROPAGATE(x)
            BACKWARD_PROPAGATE(y)
            UPDATE_WEIGHTS(x)
        CALCULATE_CROSS_ENTROPY_LOSS()

    This method as input takes X and Y
    '''
    def train(self,X,Y):
        X,Y=self.embed(X,Y)
        m=X.shape[0]
        G=[]
        gs=[]
        final_loss=0
        for step in range(self._max_steps):
            idx = np.arange(X.shape[0])
            np.random.shuffle(idx)
            X=X[idx]
            Y=Y[idx]
            percentage=1
            mse=0
            outputs=[]
            for i in range(X.shape[0]):
                output=np.array(self.forward_propagate(X[i]))
                outputs.append(output)
                self.backward_propagate(Y[i])
                self.update_params(X[i])
                mse+=(np.array(output)-Y[i])**2
                if((i/X.shape[0])*100>percentage-1):
                    progress(i/X.shape[0]+.01,title='Epoch : '+str(step+1))
                    percentage+=1
            mse/=m
            final_loss=self.cross_entropy(Y,outputs)
            print('\nEpoch:',step+1,'MSE:',np.sum(mse),'LOSS:',self.cross_entropy(Y,outputs))
        return final_loss


    '''
    Given an matrix X this Method predicts the output class for each input vector
    if Proba is set as False if Proba is True it returns the probality of each
    output class
    '''
    def predict(self,X,proba=False):
        X,_=self.embed(X,None)
        res_full=[]
        for i in range(X.shape[0]):
            res=[]
            self.forward_propagate(X[i])
            for unit in self._layers[-1]:
                res.append(unit['output'])
            if not proba:
                res_full.append(self._labels[np.argmax(np.array(res))])
            else:
                res_full.append(res)
        return np.array(res_full)

    '''
    This method Saves the Neural Network weights into pickle file
    '''
    def save_net(self,mname):
        model_data={}
        model_data['labels']=self._labels
        model_data['layers']=self._layers
        with open(mname, 'wb') as fp:
            pickle.dump(model_data, fp)

    '''
    This method loads the Neural Network weights from a pickle file
    '''
    def load_net(self,mname):
        with open (mname, 'rb') as fp:
            model_data = pickle.load(fp)
        self._labels=model_data['labels']
        self._layers=model_data['layers']


'''
End of MODEL Defination
'''

'''
TRAIN THE MODEL
'''
train=pd.read_csv('dataset/train.csv')
Y=train[['class']]
X=train[train.columns.difference(['class'])]
nn2=neural_net(85,[20,10],8,max_steps=100,lr=0.01,lamda=0.001,mom=0.2)
nn2.train(X[:10000],Y[:10000])
nn2.save_net('model.net')

'''
LOAD BEST WEIGHTS
'''
best_model=neural_net(85,[15,10],8,max_steps=100,lr=0.01,lamda=0.001,mom=0.2)
best_model.load_net('best_model.pkl')

'''
GENERATE SUBMISSION CSV
'''
test=pd.read_csv('dataset/test.csv')
test=test.reindex_axis(sorted(test.columns), axis=1)
res=nn3.predict(test)
submission=pd.DataFrame()
submission['id']=range(test.as_matrix().shape[0])
submission['predicted_class']=res
submission.to_csv('submission.csv',index=False)

'''
CROSS VALIDATION
'''
def cross_validation(X,Y,k):
    learning_rate = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    num_hidden_layers = [2, 3, 4]
    regularizer_lambda = [1e+1, 1, 1e-1]
    eval_report=[]
    idx=1
    threads=[None]*45
    for lr in learning_rate:
        for nh in num_hidden_layers:
            for lam in regularizer_lambda:
                #print(idx,lr,nh,lam)

                ln_data=X.shape[0]
                factor=int(ln_data/k)
                training_loss=0
                test_loss=0
                for i in range(k):
                    xtest=X[i*factor:(i+1)*factor]
                    xtrain=pd.concat((X[(i-1)*factor:i*factor],X[(i+1)*factor:k*factor]))
                    ytest=Y[i*factor:(i+1)*factor]
                    ytrain=pd.concat((Y[(i-1)*factor:i*factor],Y[(i+1)*factor:k*factor]))
                    model=neural_net(85,[15]*nh,10,max_steps=300,lr=lr,lamda=lam,mom=0.3)
                    training_loss+=model.train(xtrain,ytrain)
                    outputs=model.predict(xtest,proba=True)
                    _,truey=model.embed(xtest,ytest)
                    test_loss+=model.cross_entropy(truey,outputs)
                print({'No.':idx,'Hidden Layers':nh,'Learning Rate':lr,'Lambda':lam,'Training Loss':training_loss/k,'Test Loss':test_loss/k})
                eval_report.append({'No.':idx,'Hidden Layers':nh,'Learning Rate':lr,'Lambda':lam,'Training Loss':training_loss/k,'Test Loss':test_loss/k})
                idx+=1
    return eval_report

'''
CALL K FOLD CROSS VALIDATION
'''
eval_report=cross_validation(X[:1000],Y[:1000],5)
pd.DataFrame(eval_report).to_csv('eval_report.csv')
