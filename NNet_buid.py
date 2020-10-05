# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 21:46:45 2018

@author: dhy
"""

import numpy
import theano
import theano.tensor as T

GPU = True
if GPU:
    print ("Running with GPU. ")
    try: theano.config.device = 'gpu'
    except: pass 





from PIL import Image


def load_data():
    image_olive = Image.open("./Photos\\olivettifaces.jpg")
    image_dhy = []
    for i in range(10):
        image_dhy.append(Image.open('./Photos/'+str(i)+'.jpg'))
    
    img_arrayed = numpy.array(image_olive, dtype='float64') / 256   
    img_dhy_arrayed = [numpy.array(img, dtype='float64') / 256  for img in image_dhy]
    faces = numpy.zeros([410, 2679])
    

    for i in range(20):
        for j in range(20):
            faces[i*20+j] = numpy.ndarray.flatten(img_arrayed[i*57: (i+1)*57, j*47: (j+1)*47])
    for i in range(10):
        faces[400+i] = numpy.ndarray.flatten(img_dhy_arrayed[i])    
        
    labels = numpy.zeros([410])
    for i in range(410):
        labels[i] = int(i/10)
        
    
    train_data = numpy.zeros((328, 2679))
    train_label = numpy.zeros(328)
    valid_data = numpy.zeros((41, 2679))
    valid_label = numpy.zeros(41)
    test_data = numpy.zeros((41, 2679))
    test_label = numpy.zeros(41)

    for i in range(41):
        train_data[i * 8:i * 8 + 8] = faces[i * 10:i * 10 + 8]
        train_label[i * 8:i * 8 + 8] = labels[i * 10:i * 10 + 8]
        valid_data[i] = faces[i * 10 + 8]
        valid_label[i] = labels[i * 10 + 8]
        test_data[i] = faces[i * 10 + 9]
        test_label[i] = labels[i * 10 + 9]
        
        

    train_set_x = theano.shared(numpy.asarray(train_data, dtype='float64'), borrow=True)
    train_set_y = T.cast(theano.shared(numpy.asarray(train_label, dtype='float64'), borrow=True),
                         'int32')
    
    valid_set_x = theano.shared(numpy.asarray(valid_data, dtype='float64'), borrow=True)
    valid_set_y = T.cast(theano.shared(numpy.asarray(valid_label, dtype='float64'), borrow=True),
                         'int32')
    test_set_x = theano.shared(numpy.asarray(test_data, dtype='float64'), borrow=True)
    test_set_y = T.cast(theano.shared(numpy.asarray(test_label, dtype='float64'), borrow=True),
                        'int32')
    
    all_data = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return all_data

    



from theano.tensor.signal.pool import pool_2d
from theano.tensor.nnet import conv2d

class LeNetConvPoolLayer():

    def __init__(self, input, filter_shape, image_shape, poolsize=(2, 2)):


        self.W = theano.shared(numpy.random.randn(
                filter_shape[0], filter_shape[1], filter_shape[2], filter_shape[3]), 
                borrow=True)


        self.b = theano.shared(numpy.zeros(
                (filter_shape[0], ), dtype = 'float64'),
                borrow=True)

        conv_out = conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            input_shape=image_shape
        )

        pooled_out = pool_2d(
            input=conv_out,
            ws=poolsize,
            ignore_border=True
        )
        self.output = T.maximum(0, pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        self.params = [self.W, self.b]





class HiddenLayer():
    def __init__(self, input, n_in, n_out):

        self.W = theano.shared(numpy.random.randn(n_in, n_out), 
                               borrow=True)


        self.b = theano.shared(numpy.zeros((n_out, ), dtype = 'float64'),
                               borrow=True)


        unactivated_output = T.dot(input, self.W) + self.b
        self.output = T.tanh( unactivated_output)
        self.params = [self.W, self.b]

    
class Softmax():
    def __init__(self, input, n_in, n_out):

        self.W = theano.shared(value=numpy.zeros(
                (n_in, n_out), dtype = 'float64'),
                borrow=True
        )

        self.b = theano.shared(numpy.zeros((n_out, ), dtype = 'float64'),  borrow=True)
        self.y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.y_calculated = T.argmax(self.y_given_x, axis=1)
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        return T.mean(T.neq(self.y_calculated, y))

    
    
    
    
    
def save_params(param1, param2, param3, param4):
    import pickle
    write_file = open('./params.pkl', 'wb')
    pickle.dump(param1, write_file, -1)
    pickle.dump(param2, write_file, -1)
    pickle.dump(param3, write_file, -1)
    pickle.dump(param4, write_file, -1)
    write_file.close()





if __name__ == '__main__':
    all_data = load_data()
    learning_rate=0.05
    n_epochs=200
    nkerns=[5, 10]
    batch_size=41


    train_set_x, train_set_y = all_data[0]
    valid_set_x, valid_set_y = all_data[1]
    test_set_x, test_set_y = all_data[2]


    n_train_batches = train_set_x.get_value().shape[0]
    n_valid_batches = valid_set_x.get_value().shape[0]
    n_test_batches = test_set_x.get_value().shape[0]
    n_train_batches /= batch_size
    n_valid_batches /= batch_size
    n_test_batches /= batch_size
    
    
    n_train_batches = int(n_train_batches)
    n_valid_batches = int(n_valid_batches)
    n_test_batches = int(n_test_batches)


    index = T.lscalar()
    x = T.matrix()
    y = T.ivector()


    
    layer0_input = x.reshape((batch_size, 1, 57, 47))
    
    layer0 = LeNetConvPoolLayer(
        input=layer0_input,
        image_shape=(batch_size, 1, 57, 47),
        filter_shape=(nkerns[0], 1, 5, 5),
        poolsize=(2, 2)
    )


    layer1 = LeNetConvPoolLayer(
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], 26, 21),
        filter_shape=(nkerns[1], nkerns[0], 5, 5),
        poolsize=(2, 2)
    )


    layer2_input = layer1.output.flatten(2)
    
    layer2 = HiddenLayer(
        input=layer2_input,
        n_in=nkerns[1] * 11 * 8,
        n_out=2000
    )


    layer3 = Softmax(input=layer2.output, n_in=2000, n_out=41) 


    cost = layer3.negative_log_likelihood(y) + 0.01*(((layer0.W)**2).sum() + ((layer1.W)**2).sum()  + ((layer2.W)**2).sum()  + ((layer3.W)**2).sum() )



    test_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    params = layer3.params + layer2.params + layer1.params + layer0.params

    grads = T.grad(cost, params)

    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )



    best_validation_loss = 1000000
    best_epoch = 0
    test_score = 0.

    epoch = 0

    while (epoch < n_epochs):
        epoch = epoch + 1
        
        for minibatch_index in range(int(n_train_batches)):

            cost_ij = train_model(minibatch_index)

            if minibatch_index == n_train_batches-1:

                validation_losses = [validate_model(i) for i
                                     in range(int(n_valid_batches))]
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, validation error %f %%' %
                      (epoch, this_validation_loss * 100))

                if this_validation_loss < best_validation_loss:

                    best_validation_loss = this_validation_loss
                    best_epoch = epoch
                    
                    test_score = test_model(0)
                    print(('     epoch %i, test error of '
                           'best model %f %%') %
                          (epoch, test_score * 100.))


    print('Minimum validation error ： %f %% obtained at epoch %i, '
          'Where minimum test error: %f %%' %
          (best_validation_loss * 100, best_epoch, test_score * 100))
    
    save_params(layer0.params, layer1.params, layer2.params, layer3.params)


    
    
    
    
    
    
    
    
    
