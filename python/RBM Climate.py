
# coding: utf-8

# In[1]:

import os
import sys
from netCDF4 import Dataset, num2date
import numpy
from matplotlib import pyplot
#%matplotlib inline
import datetime

sys.path.append(os.path.join(os.path.expanduser("~"), "repos/deep-learning/lib"))
from dbn import DeepBeliefNet
from rbm import RBM


# In[2]:

data_dir = os.path.join(os.path.expanduser("~"), "Dropbox/research/elastic-net-downscaling/data/")
ncep_ncar = os.path.join(data_dir, "ncep_ncar")
observed_file = os.path.join(data_dir, "obs_prcp.nc")


# In[3]:

observed_data = Dataset(observed_file)
t = observed_data.variables['time'][:]
t = num2date(t, "days since 1950-01-01 00:00:00")


# In[4]:

variables = ["air"] # "pres", "rhum", "slp", "pr_wtr", "uwnd", "vwnd"]
exten = ".mon.mean.nc"
data = Dataset(os.path.join(ncep_ncar, "air"+exten))
t2 = data.variables["time"][:]
t2 = num2date(t2, "hours since 1800-01-01 00:00:0.0")
rows = (t2 >= datetime.datetime(1950, 1, 1)) & (t2 < datetime.datetime(2000, 1, 1))
numpy.reshape(data.variables['air'][rows], (600,  73*144)).shape  # confirm there are 600 rows


# In[5]:

max_lat = 75
min_lat = 0
min_lon = 180
max_lon = 315

data =  Dataset(os.path.join(ncep_ncar, "air"+exten))
lat_vals = data.variables["lat"][:]
lon_vals = data.variables["lon"][:]
lon_cols = (lon_vals >= min_lon) & (lon_vals <= max_lon)
lat_cols = (lat_vals >= min_lat) & (lat_vals <= max_lat)
nlon = sum(lon_cols)
nlat = sum(lat_cols)
cols = len(variables)*nlat*nlon
gridded = numpy.empty((sum(rows), cols))

for i, var in enumerate(variables):
    data = Dataset(os.path.join(ncep_ncar, var+exten))
    arr = numpy.reshape(data.variables[var][rows, lat_cols, lon_cols], 
                        (sum(rows), nlon*nlat))
    
    start = i*nlon*nlat
    gridded[:, start:(start+nlon*nlat)] = arr

observed_data.variables["Prcp"][1, :, :].mask.sum()


# In[6]:

shape = observed_data.variables["Prcp"][:].shape
lt = 176-1
ln = 23-1
y = observed_data.variables["Prcp"][:, lt, ln]
normalized_gridded = (gridded - gridded[:400].mean(axis=0)) / gridded[:400].std(axis=0)
#normalized_gridded = (normalized_gridded.T - normalized_gridded.T.mean(axis=0)) / normalized_gridded.T.std(axis=0)
#normalized_gridded = normalized_gridded.T

def expit(x, beta=1):
    return 1 / (1 + numpy.exp(-beta * x))

squashed_gridded = expit(normalized_gridded, beta=1)
height, bins = numpy.histogram(squashed_gridded, bins=100)
pyplot.bar(bins[:-1], height, width=1/100.)

pyplot.imshow(squashed_gridded[13].reshape(nlat,nlon))


# In[7]:

boltzmann = RBM(n_iter=100, plot_histograms=True, verbose=True, n_components=500)
boltzmann.fit(squashed_gridded)


# In[ ]:



