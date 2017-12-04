import theano
import theano.tensor as T
import numpy as np

#from theano.tensor.nnet.abstract_conv import conv2d
from theano.tensor.nnet import conv2d
from theano.tensor.nnet.bn import batch_normalization_train
from theano.tensor.nnet.bn import batch_normalization_test
from theano.tensor.signal.pool import pool_2d
from theano.ifelse import ifelse

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from six.moves import cPickle

def quantizeAct(x,B):
    return T.minimum(2.0-T.pow(2.0,1.0-B),T.round(x*T.pow(2.0,B-1.0))*T.pow(2.0,1.0-B))

def quantizeWeight(w,B):
    return T.minimum(2.0-T.pow(1.0,1.0-B),T.round(w*T.pow(2.0,B-1.0))*T.pow(2.0,1.0-B))

def quantizeNormalizedWeight(w,B,s1,s2):#please s1=1/s2 i m lazy
    return s1*T.minimum(1.0-T.pow(2.0,1.0-B),T.round(s2*w*T.pow(2.0,B-1.0))*T.pow(2.0,1.0-B))

def slopedClipping(x, m=1.0, alpha=2.0):
    return T.clip(x/m,0,alpha)

def batchNorm(x, train, gamma, beta, RM, RV, ax):
    values_train,_,_,newRM,newRV = batch_normalization_train(x,gamma,beta,axes=ax, running_mean=RM, running_var=RV)
    values = ifelse(T.neq(train,1),batch_normalization_test(x, gamma, beta, RM, RV, axes = ax),values_train)
    return values, newRM, newRV

def dropout(x, train, p_r, snrg):
    return ifelse(T.eq(train,1),T.switch(snrg.binomial(size=x.shape,p=p_r),x,0),p_r*x)

def convBNAct(x, params, train, m=1.0, alpha=2.0):
    w,b,gamma,beta,RM,RV = params
    xc = conv2d(x,w) + b.dimshuffle('x',0,'x','x')
    xbn, newRM, newRV = batchNorm(xc, train, gamma, beta, RM, RV, 'spatial')
    xact = slopedClipping(xbn, m, alpha)
    return xact, newRM, newRV

def linBNAct(x, params, train, m=1.0, alpha=2.0):
    w,b,gamma,beta,RM,RV = params
    xc = T.dot(x,w) + b
    xbn, newRM, newRV = batchNorm(xc, train, gamma, beta, RM, RV, (0,))
    xact = slopedClipping(xbn, m, alpha)
    return xact, newRM, newRV

def linOutermost(x,params):
    w,b = params
    return T.dot(x,w)+b

def initConvWeights(Nin, Nout, size, k):
    return theano.shared(np.asarray(np.random.uniform(low = -np.sqrt(6. / (Nin*size*size + Nout*size*size)), high = np.sqrt(6. / (Nin*size*size + Nout*size*size)),size=(Nout,Nin,k,k)), dtype = theano.config.floatX))

def initLinWeights(Nin, Nout):
    return theano.shared(np.asarray(np.random.uniform(low = -np.sqrt(6. / (Nout+Nin)), high = np.sqrt(6. / (Nout+Nin)),size=(Nin,Nout)), dtype = theano.config.floatX))

def initBias(Nout):
    return theano.shared(np.zeros((Nout,), dtype=theano.config.floatX))

def initBNGamma(N):
    return theano.shared(np.ones((N,), dtype=theano.config.floatX))

def initBNBeta(N):
    return theano.shared(np.zeros((N,), dtype=theano.config.floatX))

def initBNRM(N):
    return theano.shared(np.zeros((N,), dtype=theano.config.floatX))

def initBNRV(N):
    return theano.shared(np.zeros((N,), dtype=theano.config.floatX))

def initConvBN(Nin,Nout,size,k):
    params = []
    params.append(initConvWeights(Nin,Nout,size,k))
    params.append(initBias(Nout))
    params.append(initBNGamma(Nout))
    params.append(initBNBeta(Nout))
    params.append(initBNRM(Nout))
    params.append(initBNRV(Nout))
    return params

def initLinBN(Nin,Nout):
    params = []
    params.append(initLinWeights(Nin,Nout))
    params.append(initBias(Nout))
    params.append(initBNGamma(Nout))
    params.append(initBNBeta(Nout))
    params.append(initBNRM(Nout))
    params.append(initBNRV(Nout))
    return params

def initLinOutermost(Nin,Nout):
    params = []
    params.append(initLinWeights(Nin,Nout))
    params.append(initBias(Nout))
    return params

def saveParams(filename, params):
    f = open(filename, 'wb')
    for p_layer in params:
        for p in p_layer:
            cPickle.dump(p,f,protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()
    return None

def loadMNIST(filename):
    f = open(filename,'rb')
    params = []
    for i in range(4):
        current_params=[]
        current_params.append(cPickle.load(f))
        current_params.append(cPickle.load(f))
        params.append(current_params)

    return params

def loadParams(filename, NconvBN, NlinBN):
    f = open(filename, 'rb')
    params=[]
    #load convbn params: weight, bias, gamma, beta, rm, rv - 6 overall per layer
    for i in range(NconvBN):
        current_layer_params = []
        for j in range(6):
            current_layer_params.append(cPickle.load(f))
        params.append(current_layer_params)
    #load linbn params: same number
    for i in range(NlinBN):
        current_layer_params = []
        for j in range(6):
            current_layer_params.append(cPickle.load(f))
        params.append(current_layer_params)
    #load weight and bias for last layer
    last_layer_params = []
    last_layer_params.append(cPickle.load(f))
    last_layer_params.append(cPickle.load(f))
    params.append(last_layer_params)
    f.close()
    return params

def loadNormalizedParams(filename,Nlayers):
    f = open(filename, 'rb')
    params=[]
    for i in range(Nlayers):
        current_layer_params = []
        current_layer_params.append(cPickle.load(f))
        current_layer_params.append(cPickle.load(f))
        params.append(current_layer_params)
    f.close()
    return params
