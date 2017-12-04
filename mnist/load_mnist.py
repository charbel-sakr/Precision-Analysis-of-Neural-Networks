import cPickle, gzip, numpy, theano
import theano.tensor as T

def one_hot(x, n):
    """
    convert index representation to one-hot representation
    """
    x = numpy.array(x)
    assert x.ndim == 1
    return numpy.eye(n)[x]
def shared_dataset(data_xy):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX))
    shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX))
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets us get around this issue
    return shared_x, T.cast(shared_y, 'int32')

def load():
	# Load the dataset
	f = gzip.open('mnist.pkl.gz', 'rb')
	train_set, valid_set, test_set = cPickle.load(f)
	f.close()

	#test_set_x, test_set_y = shared_dataset(test_set)
	#valid_set_x, valid_set_y = shared_dataset()
	#train_set_x, train_set_y = shared_dataset(train_set)
	test_set_x, test_set_y = test_set
	valid_set_x, valid_set_y = valid_set
	train_set_x, train_set_y = train_set
	test_set_x = numpy.asarray(test_set_x, dtype=theano.config.floatX)
	test_set_y = numpy.asarray(test_set_y, dtype='int32')
	test_set_y = one_hot(test_set_y,n=10)
	valid_set_x = numpy.asarray(valid_set_x, dtype=theano.config.floatX)
	valid_set_y = numpy.asarray(valid_set_y, dtype='int32')
	valid_set_y = one_hot(valid_set_y,n=10)
	train_set_x = numpy.asarray(train_set_x, dtype=theano.config.floatX)
	train_set_y = numpy.asarray(train_set_y, dtype='int32')
	train_set_y = one_hot(train_set_y,n=10)
	return test_set_x, test_set_y.astype(dtype=theano.config.floatX), valid_set_x, valid_set_y.astype(dtype=theano.config.floatX), train_set_x, train_set_y.astype(dtype=theano.config.floatX)