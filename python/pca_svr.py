import os
import sys
import numpy
import load_data
from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVR
import argparse
from sklearn.grid_search import GridSearchCV
import pickle 

parser = argparse.ArgumentParser()
parser.add_argument("--lat", help="Training Latitude", type=float)
parser.add_argument("--lon", help="Training Longitude", type=float)
parser.add_argument("--pcanew", action='store_true', default=False)

args = parser.parse_args()

train_data = load_data.load_supervised(1950, 1985, args.lat, args.lon, 50, which='train')
test_data = load_data.load_supervised(1986, 1999, args.lat, args.lon, 50, which='test')

pca_file = os.path.join(os.path.dirname(__file__), "models/pca.pkl")
if os.path.exists(pca_file) and (not args.pcanew):
	print "Reading PCA from file"
	pca = pickle.load(open(pca_file, 'r'))
else:
	print "Fitting PCA"
	pca = RandomizedPCA()
	pca.fit(train_data.X)
	pickle.dump(pca, open(pca_file, 'w'))

## choose components which make up 98% of the variablity
components = numpy.where(pca.explained_variance_ratio_.cumsum() < 0.98)[0]
num_components = components.shape[0]
print "Number of components chosen = %i" % num_components
print "Number of observations = %i" % len(train_data.X)

x_pca_train = pca.transform(train_data.X)[:, components]
x_pca_test = pca.transform(test_data.X)[:, components]

svr_file = os.path.join(os.path.dirname(__file__), "models/svr_%2.2f_%2.2f.pkl" % (args.lat, args.lon))
if os.path.exists(svr_file):
	clf = pickle.load(open(svr_file, "r"))
else:
	## Train SVR
	print "Traing SVR"
	svr = SVR()
	params = {'C': [1.,10.], 'gamma': [1.,10.]}
	clf = GridSearchCV(svr, params, n_jobs=12)
	clf.fit(x_pca_train, train_data.y[:, 0])
	pickle.dump(clf, open(svr_file, "w"))

## Compute fit statistics
yhat = clf.predict(test_data.X)
print "Pearson Corr", pearsonr(test_data.y[:,0], yhat)
print "Spearman Corr", spearmanr(test_data.y[:,0], yhat, )
print "SSE", sum((yhat - test_data.y[:,0])**2)


