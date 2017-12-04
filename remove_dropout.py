import theano
import theano.tensor as T
import numpy as np
import layers
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import load
from theano.tensor.signal.pool import pool_2d
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
params = layers.loadParams('preretrained_87p96.save',6,2)
def feedForward(x, params):
    bn_updates = []

    l=0
    current_params = params[l]
    c1,newRM,newRV = layers.convBNAct(x,current_params,0)
    bn_updates.append((current_params[4],newRM))
    bn_updates.append((current_params[5],newRV))

    l+=1
    current_params = params[l]
    c2,newRM,newRV = layers.convBNAct(c1,current_params,0)
    bn_updates.append((current_params[4],newRM))
    bn_updates.append((current_params[5],newRV))

    p3 = pool_2d(c2,ws=(2,2),ignore_border=True)

    l+=1
    current_params = params[l]
    c4,newRM,newRV = layers.convBNAct(p3,current_params,0)
    bn_updates.append((current_params[4],newRM))
    bn_updates.append((current_params[5],newRV))

    l+=1
    current_params = params[l]
    c5,newRM,newRV = layers.convBNAct(c4,current_params,0)
    bn_updates.append((current_params[4],newRM))
    bn_updates.append((current_params[5],newRV))

    p6 = pool_2d(c5,ws=(2,2),ignore_border=True)

    l+=1
    current_params = params[l]
    c7,newRM,newRV = layers.convBNAct(p6,current_params,0)
    bn_updates.append((current_params[4],newRM))
    bn_updates.append((current_params[5],newRV))

    l+=1
    current_params = params[l]
    c8,newRM,newRV = layers.convBNAct(c7,current_params,0)
    bn_updates.append((current_params[4],newRM))
    bn_updates.append((current_params[5],newRV))

    f9 = c8.flatten(2)

    l+=1
    current_params = params[l]
    h1, newRM, newRV = layers.linBNAct(f9,current_params,0)
    bn_updates.append((current_params[4],newRM))
    bn_updates.append((current_params[5],newRV))

    l+=1
    current_params = params[l]
    h2, newRM, newRV = layers.linBNAct(h1,current_params,0)
    bn_updates.append((current_params[4],newRM))
    bn_updates.append((current_params[5],newRV))

    l+=1
    current_params = params[l]
    z = layers.linOutermost(h2,current_params)
    #
    return z,bn_updates

def dropout_removal(params):
    updates = []
    #For FC layers, I do not compensate as it turns out it is better this way
    updates.append((params[1][0],0.8*params[1][0]))
    updates.append((params[3][0],0.7*params[3][0]))
    updates.append((params[5][0],0.7*params[5][0]))
    return updates

z, _ = feedForward(x, params)
y = T.argmax(z, axis=1)
dropout_updates = dropout_removal(params)
# compile theano functions
predict = theano.function([x], y)
remove_it = theano.function([],[],updates=dropout_updates)
batch_size = 250
# test
remove_it()
running_accuracy =0.0
batches = 0
for start in range(0,10000,batch_size):
    x_batch = x_test[start:start+batch_size]
    t_batch = labels[start:start+batch_size]
    running_accuracy += np.mean(predict(x_batch) == t_batch)
    batches+=1
test_accuracy = running_accuracy/batches
print('test accuracy = ' + repr(test_accuracy))

layers.saveParams('dropout_removed_88p12.save',params)
print("Params have been saved.")
