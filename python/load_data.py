import os
import sys
from netCDF4 import Dataset, num2date
import numpy
from matplotlib import pyplot
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix

DATA_DIR = os.path.join(os.path.expanduser("~"), "data")

max_lat = 50 #75 #40
min_lat = 18  #0  #15
min_lon = 230 #180  # 220
max_lon = 290 #315 #290

variables = ["air", "pres", "pr_wtr", "rhum", "slp", "uwnd", "vwnd"]

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
    savedir = os.path.join(savedir, var)
    check_dir_exists(savedir)
    os.chdir(savedir)
    for y in years:
        url = "ftp://ftp.cdc.noaa.gov/Datasets/ncep.reanalysis/surface/%s.sig995.%i.nc" % (var, y)
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
    savedir = os.path.join(savedir, var)
    check_dir_exists(savedir)
    os.chdir(savedir)
    for y in years:
        url = "ftp://ftp.cdc.noaa.gov/Datasets/ncep.reanalysis.dailyavgs/surface/%s.sig995.%i.nc" % (var, y)
        fname = os.path.basename(url)
        if os.path.exists(fname):
            continue
        print "Downloading %s" % url
        os.system("wget %s" % url)

def load_pretraining(batchsize):
    ncep_ncar = os.path.join(DATA_DIR, "4h_ncep")

    years = range(1948, 1980)
    datalist = []

    for year in years:
        yeardata = []
        for var in variables:
            f = '%s/%s.sig995.%i.nc' % (var, var, year)
            data = Dataset(os.path.join(ncep_ncar, f))
            lat_vals = data.variables["lat"][:]
            lon_vals = data.variables["lon"][:]
            lon_cols = (lon_vals >= min_lon) & (lon_vals <= max_lon)
            lat_cols = (lat_vals >= min_lat) & (lat_vals <= max_lat)

            nlon = sum(lon_cols)
            nlat = sum(lat_cols)
            ntime = data.variables['time'][:].shape[0]
            cols = nlat*nlon
            tmp = data.variables[var][:, lat_cols, lon_cols].reshape(ntime, cols)
            yeardata.append(tmp)

        datalist.append(numpy.hstack(yeardata))

    gridded = numpy.vstack(datalist)
    X = (gridded - gridded.mean(axis=0)) / gridded.std(axis=0)

    X = X[:(batchsize * int(X.shape[0]/batchsize))]

    return DenseDesignMatrix(X=X)

def load_supervised(minyear, maxyear, lt, ln):
    observed_data_dir = os.path.join(DATA_DIR, "gridded_observed_daily")
    observed_file = os.path.join(DATA_DIR, "gridded_obs.daily.Prcp.%i.nc")
    ncep_daily_file = os.path.join(DATA_DIR, "daily_ncep", "%s", "%s.sig995.%i.nc")

    data = Dataset(observed_file % minyear)
    lat = numpy.where(data.variables["latitude"][:] < lt)
    lon = numpy.where(data.variables["latitude"][:] < ln)

    ltidx = lat[-1][-1]
    lnidx = lon[-1][-1]
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
            f = ncep_daily_file % (var, var, year)
            ncep_data = Dataset(f)
            lat_vals = ncep_data.variables["lat"][:]
            lon_vals = ncep_data.variables["lon"][:]
            lon_cols = (lon_vals >= min_lon) & (lon_vals <= max_lon)
            lat_cols = (lat_vals >= min_lat) & (lat_vals <= max_lat)

            nlon = sum(lon_cols)
            nlat = sum(lat_cols)
            ntime = ncep_data.variables['time'][:].shape[0]
            cols = nlat*nlon
            tmp = ncep_data.variables[var][:, lat_cols, lon_cols].reshape(ntime, cols)
            year_x.append(tmp)
        X.append(numpy.hstack(year_x))

    X = numpy.vstack(X)
    Y = numpy.array(Y)
    T = numpy.array(T)
    return DenseDesignMatrix(X=X, Y=Y)


if __name__ == "__main__":
    #load_pretraining()
    #load_supervised(1950, 1960, 30, 250)
    download_observed_daily()
    for var in variables:
        download_daily_data(var)
        download_4h_data(var)
