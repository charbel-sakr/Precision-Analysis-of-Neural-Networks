import theano
import theano.tensor as T
import numpy as np
import layers
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import load
from theano.tensor.signal.pool import pool_2d
from theano.tensor.nnet import conv2d

_, _, x_test, t_test = load.cifar10(dtype=theano.config.floatX, grayscale=False)
labels = np.argmax(t_test,axis=1)
x_test = x_test.reshape((x_test.shape[0],3,32,32))


# define symbolic Theano variables
x = T.tensor4()
B = T.scalar()
BA = T.fvector()
BW = T.fvector()
#ireg = T.scalar()
#selector = T.scalar()

#prepare weight
#BC architecture is 2X128C3 - MP2 - 2x256C3 - MP2 - 2x512C3 - MP2 - 2x1024FC - 10
params = layers.loadNormalizedParams('normalized_weights.save',9)
def feedForward(x, params, B, BA, BW ):
    x = layers.quantizeAct(x, B+BA.take(0))
    l=0
    current_params = params[l]
    current_params[0] = layers.quantizeNormalizedWeight(current_params[0],B+BW.take(l),16.0,0.0625)
    current_params[1] = layers.quantizeNormalizedWeight(current_params[1],B+BW.take(l),16.0,0.0625)
    c1 = conv2d(x,current_params[0]) + current_params[1].dimshuffle('x',0,'x','x')
    c1 = layers.quantizeAct(layers.slopedClipping(c1),B+BA.take(l+1))

    l+=1
    current_params = params[l]
    current_params[0] = layers.quantizeNormalizedWeight(current_params[0],B+BW.take(l),2.0,0.5)
    current_params[1] = layers.quantizeNormalizedWeight(current_params[1],B+BW.take(l),2.0,0.5)
    c2 = conv2d(c1,current_params[0]) + current_params[1].dimshuffle('x',0,'x','x')
    c2 = layers.quantizeAct(layers.slopedClipping(c2),B+BA.take(l+1))

    p3 = pool_2d(c2,ws=(2,2),ignore_border=True)

    l+=1
    current_params = params[l]
    current_params[0] = layers.quantizeNormalizedWeight(current_params[0],B+BW.take(l),4.0,0.25)
    current_params[1] = layers.quantizeNormalizedWeight(current_params[1],B+BW.take(l),4.0,0.25)
    c4 = conv2d(p3,current_params[0]) + current_params[1].dimshuffle('x',0,'x','x')
    c4 = layers.quantizeAct(layers.slopedClipping(c4),B+BA.take(l+1))

    l+=1
    current_params = params[l]
    current_params[0] = layers.quantizeNormalizedWeight(current_params[0],B+BW.take(l),2.0,0.5)
    current_params[1] = layers.quantizeNormalizedWeight(current_params[1],B+BW.take(l),2.0,0.5)
    c5 = conv2d(c4,current_params[0]) + current_params[1].dimshuffle('x',0,'x','x')
    c5 = layers.quantizeAct(layers.slopedClipping(c5),B+BA.take(l+1))

    p6 = pool_2d(c5,ws=(2,2),ignore_border=True)

    l+=1
    current_params = params[l]
    current_params[0] = layers.quantizeNormalizedWeight(current_params[0],B+BW.take(l),2.0,0.5)
    current_params[1] = layers.quantizeNormalizedWeight(current_params[1],B+BW.take(l),2.0,0.5)
    c7 = conv2d(p6,current_params[0]) + current_params[1].dimshuffle('x',0,'x','x')
    c7 = layers.quantizeAct(layers.slopedClipping(c7),B+BA.take(l+1))

    l+=1
    current_params = params[l]
    current_params[0] = layers.quantizeNormalizedWeight(current_params[0],B+BW.take(l),1.0,1.0)
    current_params[1] = layers.quantizeNormalizedWeight(current_params[1],B+BW.take(l),1.0,1.0)
    c8 = conv2d(c7,current_params[0]) + current_params[1].dimshuffle('x',0,'x','x')
    c8 = layers.quantizeAct(layers.slopedClipping(c8),B+BA.take(l+1))

    f9 = c8.flatten(2)

    l+=1
    current_params = params[l]
    current_params[0] = layers.quantizeNormalizedWeight(current_params[0],B+BW.take(l),2.0,0.5)
    current_params[1] = layers.quantizeNormalizedWeight(current_params[1],B+BW.take(l),2.0,0.5)
    h1 = T.dot(f9,current_params[0]) + current_params[1]
    h1 = layers.quantizeAct(layers.slopedClipping(h1),B+BA.take(l+1))

    l+=1
    current_params = params[l]
    current_params[0] = layers.quantizeNormalizedWeight(current_params[0],B+BW.take(l),1.0,1.0)
    current_params[1] = layers.quantizeNormalizedWeight(current_params[1],B+BW.take(l),1.0,1.0)
    h2 = layers.linOutermost(h1,current_params)
    h2 = layers.quantizeAct(layers.slopedClipping(h2),B+BA.take(l+1))

    l+=1
    current_params = params[l]
    current_params[0] = layers.quantizeNormalizedWeight(current_params[0],B+BW.take(l),1.0,1.0)
    current_params[1] = layers.quantizeNormalizedWeight(current_params[1],B+BW.take(l),1.0,1.0)
    z = layers.linOutermost(h2,current_params)
    #
    return z
z = feedForward(x, params, B, BA, BW)
y = T.argmax(z, axis=1)
# compile theano functions
predict = theano.function([x, B, BA, BW], y)

batch_size = 250
# test
BA_granular = [6.0,2.0,3.0,2.0,3.0,2.0,3.0,1.0,3.0]
BW_granular = [8.0,8.0,9.0,8.0,7.0,6.0,4.0,4.0,0.0]
BA_ICML = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
BW_ICML = [4.0,4.0,4.0,4.0,4.0,4.0,4.0,4.0,4.0,4.0,4.0]
BA_same = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
BW_same = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]

print(repr(['Granular','ICML','Same']))
for B in range(20):
    running_granular = 0.0
    running_icml = 0.0
    running_same = 0.0
    batches = 0
    for start in range(0,10000,batch_size):
        x_batch = x_test[start:start+batch_size]
        t_batch = labels[start:start+batch_size]
        running_granular+=np.mean(predict(x_batch,B+1,BA_granular,BW_granular)==t_batch)
        running_icml+=np.mean(predict(x_batch,B+1,BA_ICML,BW_ICML)==t_batch)
        running_same+=np.mean(predict(x_batch,B+1,BA_same,BW_same)==t_batch)
        batches+=1
    test_granular = running_granular/batches
    test_icml = running_icml/batches
    test_same = running_same/batches
    print(repr([1.0-test_granular,1.0-test_icml,1.0-test_same]))

