
from pylearn2.config import yaml_parse
import os
import numpy
from matplotlib import pyplot

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--mlp-yaml", help="MLP Yaml File")
parser.add_argument("--lat", help="Training Latitude", type=float)
parser.add_argument("--lon", help="Training Longitude", type=float)
parser.add_argument("--h1", help="Training Latitude", type=int)
parser.add_argument("--h2", help="Training Longitude", type=int)
parser.add_argument("--h3", help="Training Longitude", type=int)

args = parser.parse_args()
h1 = args.h1
h2 = args.h2
h3 = args.h3

if not os.path.exists("models/grbm_l1_%i.pkl" % h1):
	print "Pre-training layer 1"
	layer1_yaml = open('yamls/grbm_l1.yaml', 'r').read() % (h1, h1)
	train = yaml_parse.load(layer1_yaml)
	train.main_loop()


if not os.path.exists("models/rbm_l2_%i_%i.pkl" % (h1, h2)):
	layer2_yaml = open('yamls/rbm_l2.yaml', 'r').read() % (h1, h1, h2, h1, h2)
	train = yaml_parse.load(layer2_yaml)
	train.main_loop()


if not os.path.exists("models/rbm_l3_%i_%i.pkl" % (h2, h3)):
	layer3_yaml = open('yamls/rbm_l3.yaml', 'r').read() % (h1, h1, h2, h2, h3, h2, h3)
	train = yaml_parse.load(layer3_yaml)
	train.main_loop()

mpl_layer = open(args.mlp_yaml, 'r').read() % (args.lat, args.lon, h1, h1, h2, h2, h3, args.lat, args.lon, args.lat, args.lon)
train = yaml_parse.load(mpl_layer)
train.main_loop()


from pylearn2.utils import serial
from load_data import load_supervised
import theano

def passdata(data):
    model1_path = 'models/mlp_%2.2f_%2.2f.pkl' % (args.lat, args.lon, args.fname)
    model1 = serial.load( model1_path )

    X = model1.get_input_space().make_theano_batch()
    Y = model1.fprop(X)
    f = theano.function([X], Y)
    predicted = f(data)
    return predicted


train_data = load_supervised(1950, 1985, args.lat, args.lon, 50, which='train')
test_data = load_supervised(1986, 1999, args.lat, args.lon, 50, which='test')


y_train_hat = passdata(train_data.X)
y_train = train_data.y[:,0]

yhat = passdata(test_data.X)
y_test = test_data.y[:,0]


## RESULTS 

#pyplot.plot(y_train_hat, alpha=0.5)
#pyplot.plot(y_train, alpha=0.2)
#pyplot.savefig("train.pdf")
print "Training Correlation\n", numpy.corrcoef(y_train_hat[:,0], y_train)


#pyplot.close()
#pyplot.plot(yhat, alpha=0.5)
#pyplot.plot(y_test, alpha=0.2)
#pyplot.savefig("test.pdf")
print "Testing Correlation\n", numpy.corrcoef(yhat[:,0], y_test)


import datetime
import pandas

t0 = datetime.date(1986, 1, 1)
t1 = datetime.date(1999, 12, 31)

dates = [t0]
j = 0
while dates[j] < t1:
    j += 1
    dates += [dates[j-1] + datetime.timedelta(days=1)]

dates = dates[:y_test.shape[0]]
months = ["%i_%2i" % (d.year, d.month) for d in dates]
df = pandas.DataFrame({"months": months, "ytest": y_test, "yhat": yhat[:, 0]})

monthlymeans = pandas.pivot_table(df, values=['ytest', 'yhat'], index='months', aggfunc=numpy.mean)

print "Spearman correlation monthly means\n", monthlymeans.corr(method="spearman")
print "Pearson correlation monthly means\n", monthlymeans.corr()

