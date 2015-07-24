__author__ = 'tj'

from netCDF4 import Dataset, num2date
import os
from load_data import DATA_DIR, min_lat, max_lat, min_lon, max_lon
import numpy
import sys

# base commands
base_cmd = 'python dbm_downscaling.py --mlp-yaml %(file)s --lat %(lt)f --lon %(ln)f --h1 500 --h2 100 --h3 50'
lasso_cmd = 'python lasso.py --lat %(lt)f --lon %(ln)f'

# deep yamls
mlpdropout = 'yamls/mlp_dropout.yaml'
mlp = 'yamls/mlp.yaml'

# lets get the lat lon values
observed_data_dir = os.path.join(DATA_DIR, "gridded_observed_daily")
observed_file = os.path.join(observed_data_dir, "gridded_obs.daily.Prcp.%i.nc")

# read in an observed vile
data = Dataset(observed_file % 1950)
lat = data.variables["latitude"][:]
lon = data.variables["longitude"][:]

# compute all possible pairs
pairs = numpy.asarray([(lt, ln) for lt in lat for ln in lon])
pairsidx = numpy.asarray([(lt, ln) for lt in range(len(lat)) for ln in range(len(lon))])

# figure out which pairs have precip data
keeppairs = numpy.empty(len(pairs), dtype=bool)
keeppairs[:] = True
for j, p in enumerate(pairsidx):
    if not isinstance(data.variables["Prcp"][0, p[0], p[1]], numpy.float32):
        keeppairs[j] = False
pairs = pairs[keeppairs]
pairsidx = pairsidx[keeppairs]

# random permutation of indices
permutation = numpy.random.permutation(range(len(pairs)))

# lets model as many as possible
for i in permutation:
    lt, ln = pairs[i]
    ltidx, lnidx = pairsidx[i]
    mlpcmd = base_cmd % {'file': mlp, 'lt': lt, 'ln': ln}
    dropoutcmd = base_cmd % {'file': mlpdropout, 'lt': lt, 'ln': ln}
    lcmd = lasso_cmd % {'lt': lt, 'ln': ln}
    os.system(lcmd)
    sys.exit()


#print [data.variables["Prcp"][0, p[0], p[1]] for p in pairsidx]
#cmd = base_cmd % {"file": file, "lt": lt, "ln": ln}
