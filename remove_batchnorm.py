import theano
import theano.tensor as T
import numpy as np
import layers
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import load
from theano.tensor.signal.pool import pool_2d
from theano.tensor.nnet import conv2d

x_train, t_train, x_test, t_test = load.cifar10(dtype=theano.config.floatX, grayscale=False)
labels = np.argmax(t_test,axis=1)
x_train = x_train.reshape((x_train.shape[0],3,32,32))
x_test = x_test.reshape((x_test.shape[0],3,32,32))


# define symbolic Theano variables
x = T.tensor4()
#ireg = T.scalar()
#selector = T.scalar()

#prepare weight
#BC architecture is 2X128C3 - MP2 - 2x256C3 - MP2 - 2x512C3 - MP2 - 2x1024FC - 10
params = layers.loadParams('dropout_removed_88p12.save',6,2)
def feedForward(x, params ):

    l=0
    current_params = params[l]
    c1 = conv2d(x,current_params[0]) + current_params[1].dimshuffle('x',0,'x','x')
    c1 = layers.slopedClipping(c1)

    l+=1
    current_params = params[l]
    c2 = conv2d(c1,current_params[0]) + current_params[1].dimshuffle('x',0,'x','x')
    c2 = layers.slopedClipping(c2)

    p3 = pool_2d(c2,ws=(2,2),ignore_border=True)

    l+=1
    current_params = params[l]
    c4 = conv2d(p3,current_params[0]) + current_params[1].dimshuffle('x',0,'x','x')
    c4 = layers.slopedClipping(c4)

    l+=1
    current_params = params[l]
    c5 = conv2d(c4,current_params[0]) + current_params[1].dimshuffle('x',0,'x','x')
    c5 = layers.slopedClipping(c5)

    p6 = pool_2d(c5,ws=(2,2),ignore_border=True)

    l+=1
    current_params = params[l]
    c7 = conv2d(p6,current_params[0]) + current_params[1].dimshuffle('x',0,'x','x')
    c7 = layers.slopedClipping(c7)

    l+=1
    current_params = params[l]
    c8 = conv2d(c7,current_params[0]) + current_params[1].dimshuffle('x',0,'x','x')
    c8 = layers.slopedClipping(c8)

    f9 = c8.flatten(2)

    l+=1
    current_params = params[l]
    h1 = T.dot(f9,current_params[0]) + current_params[1]
    h1 = layers.slopedClipping(h1)

    l+=1
    current_params = params[l]
    h2 = layers.linOutermost(h1,current_params)
    h2 = layers.slopedClipping(h2)

    l+=1
    current_params = params[l]
    z = layers.linOutermost(h2,current_params)
    #
    return z

def remove_batchnorm(params):
    plain_params = []
    limits = [16,2,4,2,2,1,2,1]
    epsilon = 0.0001
    for i in range(8):
        w,b,gamma,beta,mu,variance = params[i]
        b_next = beta+(b-mu)*gamma/T.sqrt(variance+epsilon)
        b_next = T.clip(b_next,-limits[i],limits[i])
        if(i>=6):
            gamma,beta,mu,variance = (t.dimshuffle('x',0)
                    for t in (gamma,beta,mu,variance))
        else:
            gamma,beta,mu,variance = (t.dimshuffle(0,'x','x','x')
                    for t in (gamma,beta,mu,variance))
        w_next = w*gamma/(T.sqrt(variance+epsilon))
        w_next = T.clip(w_next,-limits[i],limits[i])
        plain_params.append([w_next, b_next])
    plain_params.append(params[8])
    return plain_params

plain_params = remove_batchnorm(params)
z = feedForward(x, plain_params)
y = T.argmax(z, axis=1)
# compile theano functions
predict = theano.function([x], y)

batch_size = 250
# test
running_accuracy =0.0
batches = 0
for start in range(0,10000,batch_size):
    x_batch = x_test[start:start+batch_size]
    t_batch = labels[start:start+batch_size]
    running_accuracy += np.mean(predict(x_batch) == t_batch)
    batches+=1
test_accuracy = running_accuracy/batches
print('test accuracy = ' + repr(test_accuracy))

layers.saveParams('normalized_weights.save',plain_params)
