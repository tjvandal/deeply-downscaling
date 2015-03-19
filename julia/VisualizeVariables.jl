include("ElasticNetSVM.jl")

dimlat = length(grid_lat)
dimlon = length(grid_lon)

base_dir = "/Users/tj/Desktop/elastic-net-downscaling/"
data_dir = joinpath(base_dir, "data")
ncep_ncar_path = joinpath(data_dir, "ncep_ncar")

#### READ IN GRIDDED DATA ####
variables = ["air", "pres", "rhum", "slp", "pr_wtr", "uwnd", "vwnd"]
nvar = length(variables)
exten = ".mon.mean.nc"

f1 = joinpath(ncep_ncar_path, "$(variables[1])$exten")
data = ncread(f1, variables[1])

x_plot = reshape(X_train[1,:], dimlon, dimlat, nvar)
significant_variables = reshape(nonzero_rows, dimlon, dimlat, nvar)


lon_vals = ncread(f1, "lon")
lat_vals = ncread(f1, "lat")
lon = (lon_vals .>= min_lon) & (lon_vals .<= max_lon)
lat = (lat_vals .>= min_lat) & (lat_vals .<= max_lat)

for i=1:nvar
	imagesc(significant_variables[:,:,i])
	savefig("$i.pdf")
end

#imagesc(x_plot[:, :, 1]')
#imagesc(data[:,:,1]')
# I need to take the vector that gives us info about which columns we use
# reshape this vector, plot points
# large dot to show where observation point is
