import theano
import theano.tensor as T
import numpy as np
import layers
import load_mnist
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

x_test, t_test, x_valid, t_valid, x_train, t_train = load_mnist.load()

x_train = np.concatenate((x_train,x_valid),axis=0)
t_train = np.concatenate((t_train,t_valid),axis=0)

# define symbolic Theano variables
x = T.matrix()
t = T.matrix()
lr = T.scalar()

#prepare weight
#BC architecture is 2X128C3 - MP2 - 2x256C3 - MP2 - 2x512C3 - MP2 - 2x1024FC - 10
params = layers.loadMNIST('mnist_pretrained.save')
def feedForward(x, params):
    l=0
    current_params = params[l]
    c1 = layers.linOutermost(x,current_params)
    c1 = layers.slopedClipping(c1)

    l+=1
    current_params = params[l]
    c2 = layers.linOutermost(c1,current_params)
    c2 = layers.slopedClipping(c2)

    l+=1
    current_params = params[l]
    c3 = layers.linOutermost(c2,current_params)
    c3 = layers.slopedClipping(c3)

    l+=1
    current_params = params[l]
    z = layers.linOutermost(c3,current_params)

    return z
def rm_dropout(parmas):
    updates = []
    current_layer = parmas[0]
    w = current_layer[0]
    updates.append((w,0.8*w))
    
    current_layer = parmas[1]
    w = current_layer[0]
    updates.append((w,0.75*w))
    
    current_layer = parmas[2]
    w = current_layer[0]
    updates.append((w,0.75*w))
    
    current_layer = parmas[3]
    w = current_layer[0]
    updates.append((w,0.75*w))
    
    return updates

z = feedForward(x, params)
y = T.argmax(z, axis=1)
updates = rm_dropout(params)
# compile theano functions
remove_it = theano.function([],[],updates=updates)
predict = theano.function([x], y)


batch_size = 200
# test
remove_it()
labels = np.argmax(t_test, axis=1)
running_accuracy =0.0
batches = 0
for start in range(0,10000,batch_size):
    x_batch = x_test[start:start+batch_size]
    t_batch = labels[start:start+batch_size]
    running_accuracy += np.mean(predict(x_batch) == t_batch)
    batches+=1
test_accuracy = running_accuracy/batches
print('test accuracy = ' + repr(test_accuracy))
print('test error = ' +repr(1.0-test_accuracy))
layers.saveParams('mnist_pretrained_plain.save',params)
