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
train = T.scalar()

#prepare weight
#BC architecture is 2X128C3 - MP2 - 2x256C3 - MP2 - 2x512C3 - MP2 - 2x1024FC - 10
params = []
params.append(layers.initLinOutermost(784,512))
params.append(layers.initLinOutermost(512,512))
params.append(layers.initLinOutermost(512,512))
params.append(layers.initLinOutermost(512,10))

def feedForward(x, params, train):
    snrg = RandomStreams(seed=12345)
    x = layers.dropout(x,train,0.8,snrg)
    l=0
    current_params = params[l]
    c1 = layers.linOutermost(x,current_params)
    c1 = layers.slopedClipping(c1)
    c1 = layers.dropout(c1,train,0.75,snrg)

    l+=1
    current_params = params[l]
    c2 = layers.linOutermost(c1,current_params)
    c2 = layers.slopedClipping(c2)
    c2=layers.dropout(c2,train,0.75,snrg)

    l+=1
    current_params = params[l]
    c3 = layers.linOutermost(c2,current_params)
    c3 = layers.slopedClipping(c3)
    c3 = layers.dropout(c3,train,0.75,snrg)

    l+=1
    current_params = params[l]
    z = layers.linOutermost(c3,current_params)

    return z

def mom(cost, params, learning_rate, momentum):
    updates = []

    for l in range(4):
        current_params=params[l]
        for i in range(2):#weight bias gamma beta
            p = current_params[i]
            g = T.grad(cost,p)
            mparam_i = theano.shared(np.zeros(p.get_value().shape, dtype=theano.config.floatX))
            v = momentum * mparam_i - learning_rate * (g )
            updates.append((mparam_i, v))
            updates.append((p, T.clip(p + v,-1,1)))
    return updates

z = feedForward(x, params, train)
y = T.argmax(z, axis=1)
cost = T.mean(T.nnet.categorical_crossentropy(T.nnet.softmax(z), t))
# compile theano functions
weight_updates = mom(cost, params, learning_rate=lr, momentum=0.5)
predict = theano.function([x,train], y)
train = theano.function([x, t, lr, train], cost, updates=weight_updates)


batch_size = 200
# train model
lr=0.1
for i in range(500):
    lr*=0.978
    print('\n Starting Epoch ' + repr(i))
    if ((i+1)%100==0):
        lr=0.1
    #train
    indices = np.random.permutation(60000)
    running_cost = 0.0
    batches = 0
    for start in range(0, 60000, batch_size):
        x_batch = x_train[indices[start:start + batch_size]]
        t_batch = t_train[indices[start:start + batch_size]]
        cost= train(x_batch, t_batch, lr, 1)
        running_cost = running_cost + cost
        batches = batches+1
    total_loss = running_cost/batches

    # test
    labels = np.argmax(t_test, axis=1)
    running_accuracy =0.0
    batches = 0
    for start in range(0,10000,batch_size):
        x_batch = x_test[start:start+batch_size]
        t_batch = labels[start:start+batch_size]
        running_accuracy += np.mean(predict(x_batch,0) == t_batch)
        batches+=1
    test_accuracy = running_accuracy/batches

    print('loss = ' + repr(total_loss))
    print('test accuracy = ' + repr(test_accuracy))

layers.saveParams('mnist_pretrained.save',params)
print('Params have been saved')
