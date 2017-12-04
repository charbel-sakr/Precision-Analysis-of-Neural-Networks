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
params = layers.loadMNIST('mnist_pretrained_plain.save')
def feedForward(x, params):
    evalues = []
    activations = []
    weights = []
    biases = []
    activations.append(x)
    l=0
    current_params = params[l]
    w = current_params[0]
    w_shape = T.shape(w)
    w_flattened = w.flatten()
    current_params[0] = T.reshape(w_flattened,w_shape)
    c1 = layers.linOutermost(x,current_params)
    c1 = layers.slopedClipping(c1)
    activations.append(c1)
    weights.append(w_flattened)
    biases.append(current_params[1])

    l+=1
    current_params = params[l]
    w = current_params[0]
    w_shape = T.shape(w)
    w_flattened = w.flatten()
    current_params[0] = T.reshape(w_flattened,w_shape)
    c2 = layers.linOutermost(c1,current_params)
    c2 = layers.slopedClipping(c2)
    activations.append(c2)
    weights.append(w_flattened)
    biases.append(current_params[1])

    l+=1
    current_params = params[l]
    w = current_params[0]
    w_shape = T.shape(w)
    w_flattened = w.flatten()
    current_params[0] = T.reshape(w_flattened,w_shape)
    c3 = layers.linOutermost(c2,current_params)
    c3 = layers.slopedClipping(c3)
    activations.append(c3)
    weights.append(w_flattened)
    biases.append(current_params[1])

    l+=1
    current_params = params[l]
    w = current_params[0]
    w_shape = T.shape(w)
    w_flattened = w.flatten()
    current_params[0] = T.reshape(w_flattened,w_shape)
    z = layers.linOutermost(c3,current_params)
    #z contains all numerical outputs
    weights.append(w_flattened)
    biases.append(current_params[1])

    z_fl = z.max(axis=1)
    y_fl = z.argmax(axis=1)

    for l in range(4):
        activation = activations[l]
        E=0.0
        deriv_fl = T.grad(T.sum(z_fl),activation) #sum is taken for batches shape is now batchSize x actshape
        for i in range(10):
            z_i = z.take(i,axis=1)
            deriv_i = T.grad(T.sum(z_i),activation)
            numerator = T.sqr(deriv_i - deriv_fl) #batchsize x shape
            denum = T.switch(T.eq(z_fl,z_i),1+0.0*z_i,T.sqr(z_i-z_fl)) #shape is batchsize ->need to add broadcast
            result = numerator/(denum.dimshuffle(0,'x'))
            E = E + T.sum(result)
        evalues.append(E/24.0)
        E = 0.0
        w = weights[l]
        b = biases[l]
        deriv_fl_w = T.jacobian(z_fl,w) #jacobian so shape is batchsize x shape
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

evalues= feedForward(x, params)
# compile theano function
compute = theano.function([x], evalues)


batch_size = 50
# test
x_whole = np.concatenate((x_valid,x_test),axis=0)
EA1=0
EA2=0
EA3=0
EA4=0
EW1=0
EW2=0
EW3=0
EW4=0
for i in range(0,20000,batch_size):
    Es = compute(x_whole[i:i+batch_size])
    ea1,ew1,ea2,ew2,ea3,ew3,ea4,ew4 = Es
    EA1+=ea1/batch_size
    EA2+=ea2/batch_size
    EA3+=ea3/batch_size
    EA4+=ea4/batch_size
    EW1+=ew1/batch_size
    EW2+=ew2/batch_size
    EW3+=ew3/batch_size
    EW4+=ew4/batch_size
print(repr([EA1/400,EA2/400,EA3/400,EA4/400,EW1/400,EW2/400,EW3/400,EW4/400]))

