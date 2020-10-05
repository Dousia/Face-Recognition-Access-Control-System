# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 15:47:28 2018

@author: dhy
"""

import pickle

import numpy
import theano
import theano.tensor as T




from theano.tensor.signal.pool import pool_2d
from theano.tensor.nnet import conv

from PIL import Image


def load_data_use():
    image_olive = Image.open("./Photos\olivettifaces.jpg")
    image_dhy = []
    for i in range(10):
        image_dhy.append(Image.open('./Photos\\' +str(i)+'.jpg'))
    
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
        
    faces = theano.shared(numpy.asarray(faces, dtype='float64'), borrow=True)
    labels = T.cast(theano.shared(numpy.asarray(labels, dtype='float64'), borrow=True),
                         'int32')
        
    return faces, labels




def load_params():
    filestream = open("./params.pkl", "rb")
    params0 = pickle.load(filestream)
    params1 = pickle.load(filestream)
    params2 = pickle.load(filestream)
    params3 = pickle.load(filestream)
    
    return params0, params1, params2, params3



class LeNetConvPoolLayer():

    def __init__(self, input, params, filter_shape, image_shape, poolsize=(2, 2)):

        self.W = params[0]
        self.b = params[1]

        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )

        pooled_out = pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )
        self.output = T.maximum(0, pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        self.params = [self.W, self.b]





class HiddenLayer():
    def __init__(self, input, params, n_in, n_out):

        self.W = params[0]
        self.b = params[1]

        unactivated_output = T.dot(input, self.W) + self.b
        self.output = T.tanh( unactivated_output)
        self.params = [self.W, self.b]

    
class Softmax():
    def __init__(self, input, params, n_in, n_out):

        self.W = params[0]
        self.b = params[0]
        
        self.y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.y_calculated = T.argmax(self.y_given_x, axis=1)
        self.params = [self.W, self.b]
        
    def errors(self, y):
        return T.mean(T.neq(self.y_calculated, y))
    
    def believe(self):
        return self.y_given_x[0][self.y_calculated[0]]
    
    def result(self):
        return self.y_calculated[0]



if __name__ == "__main__":
    
    params0, params1, params2, params3 = load_params()
    nkerns=[5, 10]

    x = T.matrix()    
        
    layer0_input = x.reshape((1, 1, 57, 47))
    

    layer0 = LeNetConvPoolLayer(
        input = layer0_input,
        params = params0,
        image_shape = (1, 1, 57, 47),
        filter_shape = (nkerns[0], 1, 5, 5),
        poolsize = (2, 2)
    )

    layer1 = LeNetConvPoolLayer(
        input = layer0.output,
        params = params1,
        image_shape = (1, nkerns[0], 26, 21),
        filter_shape = (nkerns[1], nkerns[0], 5, 5),
        poolsize = (2, 2)
    )

    layer2_input = layer1.output.flatten(2)
    layer2 = HiddenLayer(
        input = layer2_input,
        params = params2,
        n_in = nkerns[1] * 11 * 8,
        n_out = 2000
        )

    layer3 = Softmax(
            input = layer2.output, 
            params = params3,
            n_in = 2000, 
            n_out = 40
    )   
     
    image_face = Image.open("./disposed.jpg")
    img_arrayed = numpy.array(image_face, dtype='float64') / 256   
    face = theano.shared(img_arrayed, borrow=True)
    
    
    func_believe = theano.function([x],
                           layer3.believe())
    
    
    
    
    faces, labels = load_data_use()
    params0, params1, params2, params3 = load_params()
    nkerns=[5, 10]
    
    i = T.lscalar()
    x = T.matrix()
    faces_num = faces.get_value().shape[0]
    
        
    layer0_input = x.reshape((1, 1, 57, 47))
    

    layer0 = LeNetConvPoolLayer(
        input = layer0_input,
        params = params0,
        image_shape = (1, 1, 57, 47),
        filter_shape = (nkerns[0], 1, 5, 5),
        poolsize = (2, 2)
    )

    layer1 = LeNetConvPoolLayer(
        input = layer0.output,
        params = params1,
        image_shape = (1, nkerns[0], 26, 21),
        filter_shape = (nkerns[1], nkerns[0], 5, 5),
        poolsize = (2, 2)
    )

    layer2_input = layer1.output.flatten(2)
    layer2 = HiddenLayer(
        input = layer2_input,
        params = params2,
        n_in = nkerns[1] * 11 * 8,
        n_out = 2000
        )

    layer3 = Softmax(
            input = layer2.output, 
            params = params3,
            n_in = 2000, 
            n_out = 40
    )   
     
    
    
    func_wrong = theano.function([i],
                           layer3.errors(labels[i]),
                           givens = {
                                   x: faces[i: i+1]
                                   })
    
    func_believe = theano.function([i],
                           layer3.believe(),
                           givens = {
                                   x: faces[i: i+1]
                                   })
    

    not_match = 0
    
    for j in range(faces_num):
        wrong = func_wrong(j)
        believe = func_believe(j)
        if(wrong==1 and believe<0.8):
            not_match += 1
        
    error = not_match/faces_num
    print("Error with given parameters is:" )
    print(not_match)














