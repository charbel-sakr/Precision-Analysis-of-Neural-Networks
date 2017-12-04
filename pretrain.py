import theano
import theano.tensor as T
import numpy as np
import layers
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import load
from theano.tensor.signal.pool import pool_2d
x_train, t_train, x_test, t_test = load.cifar10(dtype=theano.config.floatX, grayscale=False)
labels_test = np.argmax(t_test,axis=1)

x_train = x_train.reshape((x_train.shape[0],3,32,32))
x_test = x_test.reshape((x_test.shape[0],3,32,32))


# define symbolic Theano variables
x = T.tensor4()
t = T.matrix()
lr = T.scalar()
train = T.scalar()

#prepare weight
#BC architecture is 2X128C3 - MP2 - 2x256C3 - MP2 - 2x512C3 - MP2 - 2x1024FC - 10
params = []
params.append(layers.initConvBN(3,64,32,5))
params.append(layers.initConvBN(64,64,28,1))
params.append(layers.initConvBN(64,64,28,1))
params.append(layers.initConvBN(64,64,14,5))
params.append(layers.initConvBN(64,64,10,1))
params.append(layers.initConvBN(64,64,10,1))
params.append(layers.initConvBN(64,64,5,5))
params.append(layers.initLinBN(64,64))
params.append(layers.initLinBN(64,64))
params.append(layers.initLinBN(64,64))
params.append(layers.initLinOutermost(64,10))

def feedForward(x, params, train):
    snrg = RandomStreams(seed=12345)
    bn_updates=[]
    l=0#64C5
    current_params = params[l]
    c,newRM,newRV = layers.convBNAct(x,current_params,train)
    bn_updates.append((current_params[4],newRM))
    bn_updates.append((current_params[5],newRV))
    c = layers.dropout(c,train,0.9,snrg)


    l+=1 #64C1
    current_params = params[l]
    c,newRM,newRV = layers.convBNAct(c,current_params,train)
    bn_updates.append((current_params[4],newRM))
    bn_updates.append((current_params[5],newRV))
    c = layers.dropout(c,train,0.9,snrg)

    l+=1 #64C1
    current_params = params[l]
    c,newRM,newRV = layers.convBNAct(c,current_params,train)
    bn_updates.append((current_params[4],newRM))
    bn_updates.append((current_params[5],newRV))

    p = pool_2d(c,ws=(2,2),ignore_border=True)

    l+=1 #64C5
    current_params = params[l]
    c,newRM,newRV = layers.convBNAct(p,current_params,train)
    bn_updates.append((current_params[4],newRM))
    bn_updates.append((current_params[5],newRV))
    c = layers.dropout(c,train,0.85,snrg)

    l+=1 #64C1
    current_params = params[l]
    c,newRM,newRV = layers.convBNAct(c,current_params,train)
    bn_updates.append((current_params[4],newRM))
    bn_updates.append((current_params[5],newRV))
    c = layers.dropout(c,train,0.85,snrg)

    l+=1 #64C1
    current_params = params[l]
    c,newRM,newRV = layers.convBNAct(c,current_params,train)
    bn_updates.append((current_params[4],newRM))
    bn_updates.append((current_params[5],newRV))

    p = pool_2d(c,ws=(2,2),ignore_border=True)

    l+=1 #64C5
    current_params = params[l]
    c,newRM,newRV = layers.convBNAct(p,current_params,train)
    bn_updates.append((current_params[4],newRM))
    bn_updates.append((current_params[5],newRV))
    c = layers.dropout(c,train,0.8,snrg)

    h = c.flatten(2)
    l+=1
    current_params = params[l]
    h, newRM, newRV = layers.linBNAct(h,current_params,train)
    bn_updates.append((current_params[4],newRM))
    bn_updates.append((current_params[5],newRV))
    h = layers.dropout(h,train,0.8,snrg)

    l+=1
    current_params = params[l]
    h, newRM, newRV = layers.linBNAct(h,current_params,train)
    bn_updates.append((current_params[4],newRM))
    bn_updates.append((current_params[5],newRV))
    h = layers.dropout(h,train,0.8,snrg)

    l+=1
    current_params = params[l]
    h, newRM, newRV = layers.linBNAct(h,current_params,train)
    bn_updates.append((current_params[4],newRM))
    bn_updates.append((current_params[5],newRV))
    h = layers.dropout(h,train,0.8,snrg)

    z = layers.linOutermost(h,params[l+1])
    return z,bn_updates

def mom(cost, params, learning_rate, momentum):
    updates = []

    for l in range(10):
        current_params=params[l]
        for i in range(4):#weight bias gamma beta
            p = current_params[i]
            g = T.grad(cost,p)
            mparam_i = theano.shared(np.zeros(p.get_value().shape, dtype=theano.config.floatX))
            v = momentum * mparam_i - learning_rate * (g )
            updates.append((mparam_i, v))
            updates.append((p, T.clip(p + v,-1,1)))

    current_params = params[10]
    for i in range(2):
        p = current_params[i]
        g = T.grad(cost,p)
        mparam_i = theano.shared(np.zeros(p.get_value().shape, dtype=theano.config.floatX))
        v = momentum * mparam_i - learning_rate * (g + 0.001*p)
        updates.append((mparam_i, v))
        updates.append((p, p + v))
    return updates

z,bn_updates = feedForward(x, params, train)
y = T.argmax(z, axis=1)
cost = T.mean(T.nnet.categorical_crossentropy(T.nnet.softmax(z), t))
# compile theano functions
weight_updates = mom(cost, params, learning_rate=lr, momentum=0.9)
predict = theano.function([x,train], y)
train = theano.function([x, t, lr, train], cost, updates=bn_updates+weight_updates)


batch_size = 200
# train model
lr=0.01
for i in range(300):
    print('\n Starting Epoch ' + repr(i))
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
        #print('minimbatch '+repr(batches)+ ' has loss '+repr(cost))
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

layers.saveParams('cifar_pretrained.save',params)
print('Params have been saved')
