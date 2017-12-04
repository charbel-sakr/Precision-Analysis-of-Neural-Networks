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
B = T.scalar()
#prepare weight
#BC architecture is 2X128C3 - MP2 - 2x256C3 - MP2 - 2x512C3 - MP2 - 2x1024FC - 10
params = layers.loadMNIST('mnist_pretrained_plain.save')
def feedForward(x, params,B):
    x = layers.quantizeAct(x,B)
    l=0
    current_params = params[l]
    current_params[0] = layers.quantizeWeight(current_params[0],B)
    current_params[1] = layers.quantizeWeight(current_params[1],B)
    c1 = layers.linOutermost(x,current_params)
    c1 = layers.slopedClipping(c1)
    c1 = layers.quantizeAct(c1,B)

    l+=1
    current_params = params[l]
    current_params[0] = layers.quantizeWeight(current_params[0],B)
    current_params[1] = layers.quantizeWeight(current_params[1],B)
    c2 = layers.linOutermost(c1,current_params)
    c2 = layers.slopedClipping(c2)
    c2 = layers.quantizeAct(c2,B)

    l+=1
    current_params = params[l]
    current_params[0] = layers.quantizeWeight(current_params[0],B)
    current_params[1] = layers.quantizeWeight(current_params[1],B)
    c3 = layers.linOutermost(c2,current_params)
    c3 = layers.slopedClipping(c3)
    c3 = layers.quantizeAct(c3,B)

    l+=1
    current_params = params[l]
    current_params[0] = layers.quantizeWeight(current_params[0],B)
    current_params[1] = layers.quantizeWeight(current_params[1],B)

    z = layers.linOutermost(c3,current_params)

    return z

z = feedForward(x, params,B)
y = T.argmax(z, axis=1)
# compile theano function
predict = theano.function([x,B], y)


batch_size = 200
# test
labels = np.argmax(t_test, axis=1)
for B in range(20):
    running_accuracy =0.0
    batches = 0
    for start in range(0,10000,batch_size):
        x_batch = x_test[start:start+batch_size]
        t_batch = labels[start:start+batch_size]
        running_accuracy += np.mean(predict(x_batch,B+1) == t_batch)
        batches+=1
    test_accuracy = running_accuracy/batches
    print(repr(1.0-test_accuracy))
