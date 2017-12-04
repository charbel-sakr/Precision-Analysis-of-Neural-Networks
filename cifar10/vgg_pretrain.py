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
t = T.matrix()
lr = T.scalar()
#ireg = T.scalar()
#selector = T.scalar()
train = T.scalar()

#prepare weight
#BC architecture is 2X128C3 - MP2 - 2x256C3 - MP2 - 2x512C3 - MP2 - 2x1024FC - 10
params =[]
params.append(layers.initConvBN(3,32,32,3))
params.append(layers.initConvBN(32,32,30,3))
params.append(layers.initConvBN(32,64,14,3))
params.append(layers.initConvBN(64,64,12,3))
params.append(layers.initConvBN(64,128,5,3))
params.append(layers.initConvBN(128,128,3,3))
params.append(layers.initLinBN(128,256))
params.append(layers.initLinBN(256,256))
params.append(layers.initLinOutermost(256,10))
def feedForward(x, params, train):
    snrg = RandomStreams(seed=12345)
    bn_updates = []

    l=0
    current_params = params[l]
    c1,newRM,newRV = layers.convBNAct(x,current_params,train)
    bn_updates.append((current_params[4],newRM))
    bn_updates.append((current_params[5],newRV))
    c1 = layers.dropout(c1,train,0.8,snrg)

    l+=1
    current_params = params[l]
    c2,newRM,newRV = layers.convBNAct(c1,current_params,train)
    bn_updates.append((current_params[4],newRM))
    bn_updates.append((current_params[5],newRV))

    p3 = pool_2d(c2,ws=(2,2),ignore_border=True)

    l+=1
    current_params = params[l]
    c4,newRM,newRV = layers.convBNAct(p3,current_params,train)
    bn_updates.append((current_params[4],newRM))
    bn_updates.append((current_params[5],newRV))
    c4 = layers.dropout(c4,train,0.7,snrg)

    l+=1
    current_params = params[l]
    c5,newRM,newRV = layers.convBNAct(c4,current_params,train)
    bn_updates.append((current_params[4],newRM))
    bn_updates.append((current_params[5],newRV))

    p6 = pool_2d(c5,ws=(2,2),ignore_border=True)

    l+=1
    current_params = params[l]
    c7,newRM,newRV = layers.convBNAct(p6,current_params,train)
    bn_updates.append((current_params[4],newRM))
    bn_updates.append((current_params[5],newRV))
    c7 = layers.dropout(c7,train,0.7,snrg)

    l+=1
    current_params = params[l]
    c8,newRM,newRV = layers.convBNAct(c7,current_params,train)
    bn_updates.append((current_params[4],newRM))
    bn_updates.append((current_params[5],newRV))

    #p9 = pool_2d(c8,ws=(2,2),ignore_border=True)
    #
    f9 = c8.flatten(2)

    l+=1
    current_params = params[l]
    h1, newRM, newRV = layers.linBNAct(f9,current_params,train)
    bn_updates.append((current_params[4],newRM))
    bn_updates.append((current_params[5],newRV))
    h1 = layers.dropout(h1,train,0.6,snrg)

    l+=1
    current_params = params[l]
    h2, newRM, newRV = layers.linBNAct(h1,current_params,train)
    bn_updates.append((current_params[4],newRM))
    bn_updates.append((current_params[5],newRV))
    h2 = layers.dropout(h2,train,0.6,snrg)

    l+=1
    current_params = params[l]
    z = layers.linOutermost(h2,current_params)
    #
    return z,bn_updates

def mom(cost, params, learning_rate, momentum):
    updates = []

    beta1 = theano.shared(np.cast[theano.config.floatX](0.9))
    beta2 = theano.shared(np.cast[theano.config.floatX](0.999))
    for current_params in params:
        p_no = 0
        for p in current_params:#weight bias gamma beta
            p_no+=1
            if(p_no==4):
                break
            g = T.grad(cost,p)

            m_adam = theano.shared(np.zeros(p.get_value().shape, dtype=theano.config.floatX))
            v_adam = theano.shared(np.zeros(p.get_value().shape, dtype=theano.config.floatX))

            m_t = 0.9*m_adam +0.1*g
            v_t = 0.999*v_adam + 0.001*T.sqr(g)

            m_hat = m_t/(1.-beta1)
            v_hat = v_t/(1.-beta2)

            updates.append((m_adam,m_t))
            updates.append((v_adam,v_t))
            v = lr*m_hat/(T.sqrt(v_hat)+0.00000001)

            updates.append((p, p - v))

    updates.append((beta1,beta1*0.9))
    updates.append((beta2,beta2*0.999))
    return updates

z, bn_updates = feedForward(x, params, train)
y = T.argmax(z, axis=1)
cost = T.mean(T.nnet.categorical_crossentropy(T.nnet.softmax(z), t))
# compile theano functions
weight_updates = mom(cost, params, learning_rate=lr, momentum=0.9)
predict = theano.function([x,train], y)
train = theano.function([x, t, lr, train], cost, updates=bn_updates+weight_updates)


batch_size = 250
# train model
lr=0.01
for i in range(600):
    print('\n Starting Epoch ' + repr(i))
    if (i==100):
        lr = 0.001
    if (i==200):
        lr = 0.0001
    if (i==400):
        lr = 0.00001
    if (i==500):
        lr = 0.000001
    #train
    indices = np.random.permutation(50000)
    running_cost = 0.0
    batches = 0
    for start in range(0, 50000, batch_size):
        x_batch = x_train[indices[start:start + batch_size]]
        t_batch = t_train[indices[start:start + batch_size]]

        #horiz flip
        coins = np.random.rand(batch_size) < 0.5
        for r in range(batch_size):
            if coins[r]:
                x_batch[r,:,:,:] = x_batch[r,:,:,::-1]

        #random crop
        padded = np.pad(x_batch,((0,0),(0,0),(4,4),(4,4)),mode='constant')
        random_cropped = np.zeros(x_batch.shape, dtype=np.float32)
        crops = np.random.random_integers(0,high=8,size=(batch_size,2))
        for r in range(batch_size):
            random_cropped[r,:,:,:] = padded[r,:,crops[r,0]:(crops[r,0]+32),crops[r,1]:(crops[r,1]+32)]

        #train
        cost= train(random_cropped, t_batch, lr, 1)
        running_cost = running_cost + cost
        batches = batches+1
    total_loss = running_cost/batches

    # test
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

layers.saveParams('vgg_from_scratch_new_gpu.save',params)
print('Params have been saved')
