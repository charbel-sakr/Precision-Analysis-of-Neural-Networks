import theano
import theano.tensor as T
import numpy as np
import layers
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import load
from theano.tensor.signal.pool import pool_2d
from theano.tensor.nnet import conv2d

x_train, _, x_test, _ = load.cifar10(dtype=theano.config.floatX, grayscale=False)
x_test = x_test.reshape((x_test.shape[0],3,32,32))
x_valid = x_train[:10000]
x_valid = x_valid.reshape((x_valid.shape[0],3,32,32))

# define symbolic Theano variables
x = T.tensor4()
#ireg = T.scalar()
#selector = T.scalar()

#prepare weight
#BC architecture is 2X128C3 - MP2 - 2x256C3 - MP2 - 2x512C3 - MP2 - 2x1024FC - 10
params = layers.loadNormalizedParams('normalized_weights.save',9)
def feedForward(x, params ):
    evalues = []
    activations = []
    weights = []
    biases = []
    activations.append(x)

    l=0
    current_params = params[l]
    w = current_params[0]
    w_shape = T.shape(w)
    wf = w.flatten()
    current_params[0] = T.reshape(wf,w_shape)
    c1 = conv2d(x,current_params[0]) + current_params[1].dimshuffle('x',0,'x','x')
    c1 = layers.slopedClipping(c1)
    activations.append(c1)
    weights.append(wf)
    biases.append(current_params[1])

    l+=1
    current_params = params[l]
    w = current_params[0]
    w_shape = T.shape(w)
    wf = w.flatten()
    current_params[0] = T.reshape(wf,w_shape)
    c2 = conv2d(c1,current_params[0]) + current_params[1].dimshuffle('x',0,'x','x')
    c2 = layers.slopedClipping(c2)
    activations.append(c2)
    weights.append(wf)
    biases.append(current_params[1])

    p3 = pool_2d(c2,ws=(2,2),ignore_border=True)

    l+=1
    current_params = params[l]
    w = current_params[0]
    w_shape = T.shape(w)
    wf = w.flatten()
    current_params[0] = T.reshape(wf,w_shape)
    c4 = conv2d(p3,current_params[0]) + current_params[1].dimshuffle('x',0,'x','x')
    c4 = layers.slopedClipping(c4)
    activations.append(c4)
    weights.append(wf)
    biases.append(current_params[1])

    l+=1
    current_params = params[l]
    w = current_params[0]
    w_shape = T.shape(w)
    wf = w.flatten()
    current_params[0] = T.reshape(wf,w_shape)
    c5 = conv2d(c4,current_params[0]) + current_params[1].dimshuffle('x',0,'x','x')
    c5 = layers.slopedClipping(c5)
    activations.append(c5)
    weights.append(wf)
    biases.append(current_params[1])

    p6 = pool_2d(c5,ws=(2,2),ignore_border=True)

    l+=1
    current_params = params[l]
    w = current_params[0]
    w_shape = T.shape(w)
    wf = w.flatten()
    current_params[0] = T.reshape(wf,w_shape)
    c7 = conv2d(p6,current_params[0]) + current_params[1].dimshuffle('x',0,'x','x')
    c7 = layers.slopedClipping(c7)
    activations.append(c7)
    weights.append(wf)
    biases.append(current_params[1])

    l+=1
    current_params = params[l]
    w = current_params[0]
    w_shape = T.shape(w)
    wf = w.flatten()
    current_params[0] = T.reshape(wf,w_shape)
    c8 = conv2d(c7,current_params[0]) + current_params[1].dimshuffle('x',0,'x','x')
    c8 = layers.slopedClipping(c8)
    activations.append(c8)
    weights.append(wf)
    biases.append(current_params[1])

    f9 = c8.flatten(2)

    l+=1
    current_params = params[l]
    w = current_params[0]
    w_shape = T.shape(w)
    wf = w.flatten()
    current_params[0] = T.reshape(wf,w_shape)
    h1 = T.dot(f9,current_params[0]) + current_params[1]
    h1 = layers.slopedClipping(h1)
    activations.append(h1)
    weights.append(wf)
    biases.append(current_params[1])

    l+=1
    current_params = params[l]
    w = current_params[0]
    w_shape = T.shape(w)
    wf = w.flatten()
    current_params[0] = T.reshape(wf,w_shape)
    h2 = layers.linOutermost(h1,current_params)
    h2 = layers.slopedClipping(h2)
    activations.append(h2)
    weights.append(wf)
    biases.append(current_params[1])

    l+=1
    current_params = params[l]
    w = current_params[0]
    w_shape = T.shape(w)
    wf = w.flatten()
    current_params[0] = T.reshape(wf,w_shape)
    z = layers.linOutermost(h2,current_params)
    weights.append(wf)
    biases.append(current_params[1])
    #
    z_fl = z.max(axis=1)
    y_fl = z.argmax(axis=1)

    for activation in activations:
        E=0.0
        deriv_fl = T.grad(T.sum(z_fl),activation)
        for i in range(10):
            z_i = z.take(i,axis=1)
            deriv_i = T.grad(T.sum(z_i),activation)
            numerator = T.sqr(deriv_i - deriv_fl)
            denum = T.switch(T.eq(z_fl,z_i),1+0.0*z_i,T.sqr(z_i-z_fl))
            numerator = numerator.flatten(2) # shape is batchsize x something big
            result = numerator/(denum.dimshuffle(0,'x'))
            E= E+T.sum(result)
        evalues.append(E/24.0)
    for l in range(9):
        w = weights[l]
        b = biases[l]
        E = 0.0
        deriv_fl_w = T.jacobian(z_fl,w)
        deriv_fl_b = T.jacobian(z_fl,b)
        for i in range(10):
            z_i = z.take(i,axis=1)
            deriv_i_w = T.jacobian(z_i,w)
            deriv_i_b = T.jacobian(z_i,b)
            numerator_w = T.sqr(deriv_i_w - deriv_fl_w)
            numerator_b = T.sqr(deriv_i_b - deriv_fl_b)
            denum = T.switch(T.eq(z_fl,z_i),1+0.0*z_i,T.sqr(z_i-z_fl))
            result_w = numerator_w/(denum.dimshuffle(0,'x'))
            result_b = numerator_b/(denum.dimshuffle(0,'x'))
            E = E+T.sum(result_w)
            E = E+T.sum(result_b)
        evalues.append(E/24.0)
    return evalues

evalues = feedForward(x, params)
# compile theano functions
compute = theano.function([x], evalues)

batch_size = 50
x_whole = np.concatenate((x_valid,x_test),axis=0)
EA1=0
EA2=0
EA3=0
EA4=0
EA5=0
EA6=0
EA7=0
EA8=0
EA9=0
EW1=0
EW2=0
EW3=0
EW4=0
EW5=0
EW6=0
EW7=0
EW8=0
EW9=0
for i in range(0,20000,batch_size):
    Es = compute(x_whole[i:i+batch_size])
    ea1,ea2,ea3,ea4,ea5,ea6,ea7,ea8,ea9,ew1,ew2,ew3,ew4,ew5,ew6,ew7,ew8,ew9 = Es
    EA1+=ea1/batch_size
    EA2+=ea2/batch_size
    EA3+=ea3/batch_size
    EA4+=ea4/batch_size
    EA5+=ea5/batch_size
    EA6+=ea6/batch_size
    EA7+=ea7/batch_size
    EA8+=ea8/batch_size
    EA9+=ea9/batch_size
    EW1+=256.0*ew1/batch_size
    EW2+=4.0*ew2/batch_size
    EW3+=16.0*ew3/batch_size
    EW4+=4.0*ew4/batch_size
    EW5+=4.0*ew5/batch_size
    EW6+=ew6/batch_size
    EW7+=4.0*ew7/batch_size
    EW8+=ew8/batch_size
    EW9+=ew9/batch_size
print(repr([EA1/400,EA2/400,EA3/400,EA4/400,EA5/400,EA6/400,EA7/400,EA8/400,EA9/400,EW1/400,EW2/400,EW3/400,EW4/400,EW5/400,EW6/400,EW7/400,EW8/400,EW9/400]))

