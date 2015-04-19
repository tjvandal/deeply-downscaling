
from pylearn2.config import yaml_parse
import os
import numpy
from matplotlib import pyplot

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--mlp_yaml", help="MLP Yaml File")
parser.add_argument("--lat", help="Training Latitude", type=float)
parser.add_argument("--lon", help="Training Longitude", type=float)
args = parser.parse_args()


# In[2]:

#layer1_yaml = open('grbm_l1.yaml', 'r').read()
#train = yaml_parse.load(layer1_yaml)
#train.main_loop()


# In[5]:

#layer2_yaml = open('rbm_l2.yaml', 'r').read()
#train = yaml_parse.load(layer2_yaml)
#train.main_loop()


# In[6]:

#layer3_yaml = open('rbm_l3.yaml', 'r').read()
#train = yaml_parse.load(layer3_yaml)
#train.main_loop()


# In[8]:

layer3_yaml = open(args.mlp_yaml, 'r').read() % (args.lat, args.lon, args.lat, args.lon, args.lat, args.lon)
train = yaml_parse.load(layer3_yaml)
train.main_loop()


# In[9]:

from pylearn2.utils import serial
from load_data import load_supervised
import theano


# In[10]:

def passdata(data):
    model1_path = 'mlp_dropout.pkl'
    model1 = serial.load( model1_path )

    X = model1.get_input_space().make_theano_batch()
    Y = model1.fprop(X)
    f = theano.function([X], Y )
    predicted = f(data)
    return predicted


# In[11]:

train_data = load_supervised(1950, 1980, 42, 250, 50, which='train')
test_data = load_supervised(1981, 1999, 42, 250, 50, which='test')


# In[12]:

y_train_hat = passdata(train_data.X)
y_train = train_data.y[:,0]

yhat = passdata(test_data.X)
y_test = test_data.y[:,0]


# In[13]:

numpy.mean((y_train_hat - y_train)**2)

pyplot.plot(y_train_hat, alpha=0.5)
pyplot.plot(y_train, alpha=0.2)
pyplot.show()
numpy.corrcoef(y_train_hat[:,0], y_train)


# In[14]:

numpy.mean((yhat - y_test)**2)

pyplot.plot(yhat, alpha=0.5)
pyplot.plot(y_test, alpha=0.2)
pyplot.show()
numpy.corrcoef(yhat[:,0], y_test)


