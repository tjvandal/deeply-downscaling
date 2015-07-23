import os
import sys
import numpy
from matplotlib import pyplot
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
import pickle

DATA_DIR = os.path.join(os.path.expanduser("~"), "data")

max_lat = 50 #75 #40
min_lat = 24  #0  #15
min_lon = 220 #180  # 220
max_lon = 300 #315 #290

variables = [["air", "sig995"], ["rhum", "sig995"], ["pr_wtr", "eatm"],
             ["uwnd", "sig995"], ["vwnd", "sig995"], ['pres', 'sfc']]

def check_dir_exists(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

## download y variable, daily prcp
def download_observed_daily():
    years = range(1950, 2000)
    savedir = os.path.join(DATA_DIR, "gridded_observed_daily/")
    check_dir_exists(savedir)
    os.chdir(savedir)
    for y in years:
        ## observed prcp
        url = "http://hydro.engr.scu.edu/files/gridded_obs/daily/ncfiles/gridded_obs.daily.Prcp.%i.nc.gz" % y
        fname = os.path.basename(url)
        if os.path.exists(fname):
            continue
        print "Downloading %s" % url
        os.system("wget %s" % url)
    os.system("gunzip *.gz")

## reanalysis 4h data
def download_4h_data(var):
    years = range(1948, 2015)
    savedir = os.path.join(DATA_DIR, "4h_ncep" )
    check_dir_exists(savedir)
    savedir = os.path.join(savedir, var[0])
    check_dir_exists(savedir)
    os.chdir(savedir)
    for y in years:
        url = "ftp://ftp.cdc.noaa.gov/Datasets/ncep.reanalysis/surface/%s.%s.%i.nc" % (var[0], var[1], y)
        fname = os.path.basename(url)
        if os.path.exists(fname):
            continue
        print "Downloading %s" % url
        os.system("wget %s" % url)


## reanalysis daily data
def download_daily_data(var):
    years = range(1948, 2015)
    savedir = os.path.join(DATA_DIR, "daily_ncep" )
    check_dir_exists(savedir)
    savedir = os.path.join(savedir, var[0])
    check_dir_exists(savedir)
    os.chdir(savedir)
    for y in years:
        url = "ftp://ftp.cdc.noaa.gov/Datasets/ncep.reanalysis.dailyavgs/surface/%s.%s.%i.nc" % (var[0], var[1], y)
        fname = os.path.basename(url)
        if os.path.exists(fname):
            continue
        print "Downloading %s" % url
        os.system("wget %s" % url)

def load_pretraining(minyear, maxyear, batchsize):
    ncep_ncar = os.path.join(DATA_DIR, "4h_ncep")
    pretraining_file = os.path.join(DATA_DIR, 'downscaling-deeplearning', "pretrain.pkl")
    print pretraining_file
    if os.path.exists(pretraining_file):
        print "reading from pretraining file"
        return pickle.load(open(pretraining_file, 'r'))

    from netCDF4 import Dataset, num2date
    years = range(minyear, maxyear+1)
    datalist = []

    for year in years:
        yeardata = []
        for var in variables:
            f = '%s/%s.%s.%i.nc' % (var[0], var[0], var[1], year)
            data = Dataset(os.path.join(ncep_ncar, f))
            lat_vals = data.variables["lat"][:]
            lon_vals = data.variables["lon"][:]
            lon_cols = (lon_vals >= min_lon) & (lon_vals <= max_lon)
            lat_cols = (lat_vals >= min_lat) & (lat_vals <= max_lat)

            nlon = sum(lon_cols)
            nlat = sum(lat_cols)
            ntime = data.variables['time'][:].shape[0]
            cols = nlat*nlon
            tmp = data.variables[var[0]][:, lat_cols, lon_cols].reshape(ntime, cols)
            yeardata.append(tmp)

        datalist.append(numpy.hstack(yeardata))

    gridded = numpy.vstack(datalist)
    X = (gridded - gridded.mean(axis=0)) / gridded.std(axis=0)
    print "Shape of features:", X.shape
    X = X[:(batchsize * int(X.shape[0]/batchsize))]
    pickle.dump(DenseDesignMatrix(X=X), open(pretraining_file, "w"))
    return DenseDesignMatrix(X=X)

def load_supervised(minyear, maxyear, lt, ln, batchsize, which='train'):
    if ln > 180:
        ln = ln - 360
    observed_data_dir = os.path.join(DATA_DIR, "gridded_observed_daily")
    observed_file = os.path.join(observed_data_dir, "gridded_obs.daily.Prcp.%i.nc")
    ncep_daily_file = os.path.join(DATA_DIR, "daily_ncep", "%s", "%s.%s.%i.nc")

    supervised_file = os.path.join(DATA_DIR, 'downscaling-deeplearning', "%s_%i_%i_%2.2f_%2.2f.pkl" % (which, minyear, maxyear, lt, ln))
    transform_file = os.path.join(DATA_DIR, 'downscaling-deeplearning', "trainsform_%2.2f_%2.2f.pkl" % (lt, ln))
    print supervised_file
    if os.path.exists(supervised_file):
        print "reading from %s file" % which
        return pickle.load(open(supervised_file, 'r'))

    from netCDF4 import Dataset, num2date
    data = Dataset(observed_file % minyear)
    lat = numpy.where(data.variables["latitude"][:] <= lt)
    lon = numpy.where(data.variables["longitude"][:] <= ln)

    ltidx = lat[-1][-1]
    lnidx = lon[-1][-1]

    print "Lat in %2.4f with Lon %2.4f" % (lt, ln)
    print "Training with Lat=%3.4f and Lon=%3.4f" % (data.variables["latitude"][ltidx], data.variables["longitude"][lnidx])

    Y = []
    T = []
    X = []
    for year in range(minyear, maxyear+1):
        data = Dataset(observed_file % year)
        y = data.variables["Prcp"][:, ltidx, lnidx]
        T += data.variables["time"][:].tolist()
        Y += y.tolist()

        year_x = []
        for var in variables:
            f = ncep_daily_file % (var[0], var[0], var[1], year)
            ncep_data = Dataset(f)
            lat_vals = ncep_data.variables["lat"][:]
            lon_vals = ncep_data.variables["lon"][:]
            lon_cols = (lon_vals >= min_lon) & (lon_vals <= max_lon)
            lat_cols = (lat_vals >= min_lat) & (lat_vals <= max_lat)

            nlon = sum(lon_cols)
            nlat = sum(lat_cols)
            ntime = ncep_data.variables['time'][:].shape[0]
            cols = nlat*nlon
            tmp = ncep_data.variables[var[0]][:, lat_cols, lon_cols].reshape(ntime, cols)
            year_x.append(tmp)
        X.append(numpy.hstack(year_x))

    X = numpy.vstack(X)
    Y = numpy.reshape(numpy.array(Y), (len(Y),1))
    T = numpy.array(T)

    if which != 'train':
        if not os.path.exists(transform_file):
            print "Warning: Transformation file does not exist. Will use current dataset to create file"
            transform = {'mu': X.mean(axis=0), 'std': X.std(axis=0)}
            pickle.dump(transform, open(transform_file, "w"))
        transform = pickle.load(open(transform_file, 'r'))
    else:
        transform = {'mu': X.mean(axis=0), 'std': X.std(axis=0)}
        pickle.dump(transform, open(transform_file, "w"))

    X = (X - transform['mu']) / transform['std']

    X = X[:(batchsize * int(X.shape[0]/batchsize))]
    Y = Y[:(batchsize * int(len(Y)/batchsize))]
    out = DenseDesignMatrix(X=X, y=numpy.log(Y+1))
    pickle.dump(out, open(supervised_file, "w"))
    return out


if __name__ == "__main__":
    #download_observed_daily()
    for var in variables:
        download_daily_data(var)
        download_4h_data(var)

    #load_pretraining(1950, 1980, 50)
    data = load_supervised(1950, 1980, 42, 250, 50, which='train')
    #load_supervised(1981, 1999, 42, 250, 50, which='test')
    from matplotlib import pyplot
    pyplot.subplot(3, 1, 1)
    pyplot.hist(data.y, bins=20)
    pyplot.subplot(3, 1, 2)
    pyplot.hist(numpy.log(data.y+1), bins=20)
    pyplot.subplot(3, 1, 3)
    pyplot.hist(data.y**(0.10), bins=20)
    pyplot.show()