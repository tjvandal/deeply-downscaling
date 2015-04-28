import os
import sys
import numpy
import load_data
import argparse
import pickle
from scipy.stats.stats import pearsonr, spearmanr
from matplotlib import pyplot
from pylearn2.utils import serial
import theano
import seaborn

lat = 42.36
lon = 289.0
train_data = load_data.load_supervised(1950, 1985, lat, lon, 50, which='train')
test_data = load_data.load_supervised(1986, 1999, lat, lon, 50, which='test')

print "Training data size:", train_data.X.shape
print "Test data size:", test_data.X.shape


mlp_model = serial.load("models/mlp_dropout_%2.2f_%2.2f.pkl" % (lat, lon))
lasso_model = pickle.load(open("models/lasso_%2.2f_%2.2f.pkl" % (lat, lon), 'r'))
pca_model = pickle.load(open("models/pca.pkl", 'r'))
svr_model = pickle.load(open("models/svr_%2.2f_%2.2f.pkl" % (lat, lon), 'r'))

components = numpy.where(pca_model.explained_variance_ratio_.cumsum() < 0.98)[0]

def mlp_predict(data, model):
    X = model.get_input_space().make_theano_batch()
    Y = model.fprop(X)
    f = theano.function([X], Y)
    predicted = f(data)
    return predicted

yhat_mlp = mlp_predict(test_data.X, mlp_model)
yhat_svr = svr_model.predict(test_data.X[:, components])
yhat_lasso = lasso_model.predict(test_data.X)

ymax = max([max(test_data.y), max(yhat_mlp), max(yhat_lasso)])

mse_mpl = numpy.sum((test_data.y - yhat_mlp)**2) / len(test_data.y)
mse_svr = sum((test_data.y - yhat_svr)**2) / len(test_data.y)
mse_lasso = numpy.sum((test_data.y - numpy.expand_dims(yhat_lasso, 1))**2) / len(test_data.y)

print "\nTest Mean =", numpy.mean(test_data.y)
print "MLP Mean =", numpy.mean(yhat_mlp)
print "Lasso Mean =", numpy.mean(yhat_lasso)

print "\nTest Var =", numpy.var(test_data.y)
print "MLP Var =", numpy.var(yhat_mlp)
print "Lasso Var =", numpy.var(yhat_lasso)

print "\nMLP MSE =", mse_mpl
print "Lasso MSE =", mse_lasso

print "\nPearson Corr"
print "MLP Corr =", pearsonr(yhat_mlp[:, 0], test_data.y[:, 0])
print "Lasso Corr =", pearsonr(yhat_lasso, test_data.y[:, 0])

print "\nSpearman Corr"
print "MLP Corr =", spearmanr(yhat_mlp[:,0], test_data.y[:, 0])
print "Lasso Corr =", spearmanr(yhat_lasso, test_data.y[:, 0])

import datetime
import pandas

t0 = datetime.date(1986, 1, 1)
t1 = datetime.date(1999, 12, 31)

dates = [t0]
j = 0
while dates[j] < t1:
    j += 1
    dates += [dates[j-1] + datetime.timedelta(days=1)]

dates = dates[:test_data.y.shape[0]]


pyplot.figure(figsize=(9, 10))
pyplot.subplot(3, 1, 1)
pyplot.plot(dates, test_data.y)
pyplot.title("Test Data - Daily from 1986 to 1999")

pyplot.subplot(3, 1, 2)
pyplot.plot(dates, yhat_mlp)
pyplot.title("Deep Learning Predicted")
pyplot.text(dates[1], 50, "Pearson: %2.4f" % pearsonr(yhat_mlp[:, 0], test_data.y[:, 0])[0])
pyplot.text(dates[1], 40, "Spearman: %2.4f" % spearmanr(yhat_mlp[:, 0], test_data.y[:, 0])[0])

pyplot.subplot(3, 1, 3)
pyplot.plot(dates, yhat_lasso)
pyplot.title("Lasso Predicted")
pyplot.text(dates[1], 30, "Pearson: %2.4f" % pearsonr(yhat_lasso, test_data.y[:, 0])[0])
pyplot.text(dates[1], 25, "Spearman: %2.4f" % spearmanr(yhat_lasso, test_data.y[:, 0])[0])

pyplot.tight_layout()
pyplot.savefig("/Users/tj/Dropbox/coursework/eece7313/project/timeseries-prediction.pdf")
pyplot.show()



## Monthly Prediction

months = ["%i_%2i" % (d.year, d.month) for d in dates]
df = pandas.DataFrame({"months": months, "ytest": test_data.y[:, 0],
                       "mlp": yhat_mlp[:, 0], "lasso": yhat_lasso, "svr": yhat_svr})

monthlymeans = pandas.pivot_table(df, values=['ytest', 'mlp', 'lasso', "svr"],
                                  index='months', aggfunc=numpy.max)


monthly_mlp_mse = numpy.mean((monthlymeans.ytest - monthlymeans.mlp)**2)
monthly_lasso_mse = numpy.mean((monthlymeans.ytest - monthlymeans.lasso)**2)

print "MLP Monthly MSE =", monthly_mlp_mse
print "Lasso Monthly MSE =", monthly_lasso_mse

print "Monthly Means =", monthlymeans.mean()

print "Monthly Var =", monthlymeans.var()

print "Spearman correlation monthly means\n", monthlymeans.corr(method="spearman")
print "Pearson correlation monthly means\n", monthlymeans.corr()



pyplot.close()

#pyplot.subplot(3, 1, 1)
monthlymeans['ytest'].plot()
#pyplot.subplot(3, 1, 2)
monthlymeans['mlp'].plot(alpha=0.5)
#pyplot.subplot(3, 1, 3)
#monthlymeans['lasso'].plot()
#pyplot.show()