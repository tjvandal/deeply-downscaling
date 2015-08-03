import os
import sys
import numpy
import load_data
import argparse
import pickle
from scipy.stats.stats import pearsonr, spearmanr, kendalltau
from matplotlib import pyplot
from pylearn2.utils import serial
import theano
import seaborn
import pandas


MLP_DROPOUT_FILE = "/scratch/vandal.t/downscaling-deeplearning/mlp_dropout_%2.2f_%2.2f.pkl"
MLP_FILE = "/scratch/vandal.t/downscaling-deeplearning/mlp_%2.2f_%2.2f.pkl"
LASSO_FILE = "/scratch/vandal.t/downscaling-deeplearning/lasso_%2.2f_%2.2f.pkl"

class Compare:
    def __init__(self, lat, lon):
        if not os.path.exists(MLP_FILE % (lat, lon)):
            raise OSError
        if not os.path.exists(LASSO_FILE % (lat, lon)):
            raise OSError

        self.mlp_dropout_model = serial.load(MLP_DROPOUT_FILE % (lat, lon))
        self.mlp_model = serial.load(MLP_FILE % (lat, lon))
        self.lasso_model = pickle.load(open(LASSO_FILE % (lat, lon), 'r'))
        self.test_data = load_data.load_supervised(1986, 1999, lat, lon, 50, which='test')
        self.lat = lat
        self.lon = lon

    def get_results(self):
        results = []
        results.append(self.get_mlp_metrics(self.mlp_dropout_model, "mlp_dropout"))
        results.append(self.get_mlp_metrics(self.mlp_model, "mlp"))
        results.append(self.get_lasso_metrics())
        return results

    def get_metrics(self, y, yhat, name):
        mse = self.compute_mse(y, yhat)
        pearson = pearsonr(y, yhat)[0][0]
        kendall = kendalltau(y, yhat)[0]
        spearman = spearmanr(y, yhat)[0]
        return {"lat": self.lat, "lon": self.lon, "model": name, "mse": mse,
                "pearson": pearson, "kendall": kendall, "spearman": spearman}


    def get_mlp_metrics(self, model, name):
        yhat = self.mlp_predict(self.test_data.X, model)
        y = self.test_data.y
        return self.get_metrics(y, yhat, name)

    def get_lasso_metrics(self):
        y = self.test_data.y
        yhat = self.lasso_model.predict(self.test_data.X)[:, numpy.newaxis]
        return self.get_metrics(y, yhat, "lasso")

    def compute_mse(self, y, yhat):
        return numpy.mean((y-yhat)**2)


    def mlp_predict(self, data, model):
        X = model.get_input_space().make_theano_batch()
        Y = model.fprop(X)
        f = theano.function([X], Y)
        predicted = f(data)
        return predicted



if __name__ == "__main__":
    store_results = []
    dropout_files = [f for f in os.listdir(os.path.dirname(MLP_DROPOUT_FILE)) if "mlp_dropout" in f]
    for f in dropout_files:
        _, _, lat, lon = f.split("_")
        lat = float(lat)
        lon = float(lon.split(".pkl")[0])
        try:
            c = Compare(lat, lon)
            store_results += c.get_results()
        except OSError:
            print "Could not compare lat=%2.2f lon=%2.2f" % (lat, lon)
    print pandas.DataFrame(store_results)