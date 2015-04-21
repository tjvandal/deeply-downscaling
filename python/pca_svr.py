import os
os.chdir("/Users/tj/repos/lasso-downscaling/python")
import sys
sys.path.append("/Users/tj/repos/lasso-downscaling/python")
import numpy
import load_data
from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVR
train_data = load_data.load_supervised(1950, 1980, 42, 250, 50, which='train')
test_data = load_data.load_supervised(1981, 1999, 42, 250, 50, which='test')

print "Fitting PCA"
pca = RandomizedPCA()
pca.fit(train_data.X)

## choose components which make up 98% of the variablity
components = numpy.where(pca.explained_variance_ratio_.cumsum() < 0.98)[0]
num_components = components.shape[0]
print "number of components chosen = %i" % num_components

x_pca_train = pca.transform(train_data.X)[:, components]
x_pca_test = pca.transform(test_data.X)[:, components]

## Train SVR
svr = SVR()
svr.fit(x_pca_train, train_data.y[:, 0])
svr.score(x_pca_test, test_data.y[:, 0])
svr.score(x_pca_train, train_data.y[:, 0])
