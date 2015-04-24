import os
import sys
import numpy
import load_data
import argparse
import pickle 
from scipy.stats.stats import pearsonr, spearmanr
from sklearn.linear_model import Lasso, ElasticNet, LassoCV, LassoLarsCV

parser = argparse.ArgumentParser()
parser.add_argument("--lat", help="Training Latitude", type=float)
parser.add_argument("--lon", help="Training Longitude", type=float)

args = parser.parse_args()

train_data = load_data.load_supervised(1950, 1985, args.lat, args.lon, 50, which='train')
test_data = load_data.load_supervised(1986, 1999, args.lat, args.lon, 50, which='test')

lasso_file = os.path.join(os.path.dirname(__file__), "models/lasso_%2.2f_%2.2f.pkl" % (args.lat, args.lon))
if os.path.exists(lasso_file):
	print "Reading PCA from file"
	L = pickle.load(open(lasso_file, 'r'))
else:
	print "Fitting Lasso"
	L = LassoLarsCV(cv=5)
	L.fit(train_data.X, train_data.y[:,0])
	pickle.dump(L, open(lasso_file, 'w'))


## Print Fit stats
print "Alpha", L.alpha_ 
print "Training Pearson Corr:", pearsonr(train_data.y[:,0], L.predict(train_data.X))
print "Training Spearman Corr:", spearmanr(train_data.y[:,0], L.predict(train_data.X))

yhat = L.predict(test_data.X)
print "Pearson Corr", pearsonr(test_data.y[:,0], yhat)
print "Spearman Corr", spearmanr(test_data.y[:,0], yhat)
print "SSE", sum((yhat - test_data.y[:,0])**2)


## Compute monthly data
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
months = ["%i_%2i" % (d.year, d.month) for d in dates]
df = pandas.DataFrame({"months": months, "ytest": test_data.y[:,0], "yhat": yhat})
monthlymeans = pandas.pivot_table(df, values=['ytest', 'yhat'], index='months', aggfunc=numpy.mean)

print "Monthly Means Pearson\n", df.corr()
print "Monthly Means Pearson\n", df.corr(method="spearman")


