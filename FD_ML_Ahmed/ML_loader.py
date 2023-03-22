"""
ML_loader
~~~~~~~~~~~~
A library to load the ML data.  For details of the data
structures that are returned, see the doc strings for ``load_data``
and ``load_data_wrapper``.  In practice, ``load_data_wrapper`` is the
function usually called by our neural network code.
"""

#### Libraries
# Standard library
import pickle as cPickle
import gzip
import csv


# Third-party libraries
import numpy as np

def load_data():
    """Return the ML data as a tuple containing the training data,
    the validation data, and the test data.
    The ``training_data`` is returned as a tuple with two entries.
    The first entry contains the actual ADC Inputs.  This is a
    numpy ndarray with 6144 entries.  Each entry is, in turn, a
    numpy ndarray with 280 values, representing the 2 * 60 * 2 = 240
    numbers in a single ML iteration.
    The second entry in the ``training_data`` tuple is a numpy ndarray
    containing 1,536 entries.  Those entries are just the digit
    values (0...9) for the corresponding images contained in the first
    entry of the tuple.
    The ``validation_data`` and ``test_data`` are similar, except
    each contains only 1,680 images.
    This is a nice data format, but for use in neural networks it's
    helpful to modify the format of the ``training_data`` a little.
    That's done in the wrapper function ``load_data_wrapper()``, see
    below.
    """
    #f = open('testDataexport_train.csv', 'r')
    #training_data = (f.read())
    #f.close()
    
    f=open('testDataexport_train.csv','r')
    dataString=f.read()
    training_data=dataString.split("\n")
    for i in range (0, len(training_data),1):
        training_data[i]=training_data[i].replace(",","  ")
        training_data[i]=float(training_data[i])
        
    
 
    f = open('testDataexport_validate.csv', 'r')
    validation_data = f.read()
    f.close()
    f = open('testDataexport_validate.csv', 'r')
    test_data = f.read()
    f.close()
    
    f = open('testDataresults_train.csv', 'r')
    training_results = f.read()
    f.close()
    
    f = open('testDataexport_validate.csv', 'r')
    validation_results = f.read()
    f.close()
    f = open('testDataexport_validate.csv', 'r')
    test_results = f.read()
    f.close()
    

    #training_data, validation_data, test_data = cPickle.load(f, encoding='iso-8859-1')
   
    #training_data, validation_data, test_data = cPickle.load(f,encoding = 'latin1')
    return (training_data, validation_data, test_data, training_results, validation_results, test_results)

    
def load_data_wrapper():
    """
    Return a tuple containing ``(training_data, validation_data,
    test_data)``. Based on ``load_data``, but the format is more
    convenient for use in our implementation of neural networks.
    In particular, ``training_data`` is a list containing 50,000
    2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
    containing the input image.  ``y`` is a 10-dimensional
    numpy.ndarray representing the unit vector corresponding to the
    correct digit for ``x``.
    ``validation_data`` and ``test_data`` are lists containing 10,000
    2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
    numpy.ndarry containing the input image, and ``y`` is the
    corresponding classification, i.e., the digit values (integers)
    corresponding to ``x``.
    Obviously, this means we're using slightly different formats for
    the training data and the validation / test data.  These formats
    turn out to be the most convenient for use in our neural network
    code.
    """
    tr_d, va_d, te_d, tr_r, va_r, te_r = load_data()
    #f = open('test.csv', 'w')
    #training_data, validation_data, test_data = cPickle.load(f, encoding='iso-8859-1')
    #f.write(tr_d)
    #training_data, validation_data, test_data = cPickle.load(f,encoding = 'latin1')
    #f.close()
    #print(tr_d[0].shape)

   # training_inputs = [np.reshape(x, (240, 1)) for x in tr_d[0]]
    training_inputs = [np.reshape(tr_d, (240,1))]

   # training_results = [vectorized_result(y) for y in tr_d[1]]
    training_results = [np.reshape(tr_r, (20,1))]

    training_data = zip(training_inputs, training_results)
    #validation_inputs = [np.reshape(x, (240, 1)) for x in va_d[0]]
    validation_inputs = va_d
    validation_results= va_r

    #validation_data = zip(validation_inputs, va_d[1])
    validation_data = zip(validation_inputs, validation_results)
    test_inputs = te_d
    test_results = te_r
   # test_inputs = [np.reshape(x, (240, 1)) for x in te_d[0]]
   # test_data = zip(test_inputs, te_d[1])
    test_data = zip(test_inputs, test_results)

    return (training_data, validation_data, test_data)
    
def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


     