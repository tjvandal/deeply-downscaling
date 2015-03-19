using NetCDF
using Dates

function readdata(min_lat=0, max_lat=360, min_lon=0, max_lon=360)

	init_date = Date(1800)
	min_year = 1950
	max_year = 1999

	base_dir = dirname(dirname(@__FILE__()))
	data_dir = joinpath(base_dir, "data")
	gridded_data_file = joinpath(data_dir, "obs_prcp.nc")
	ncep_ncar_path = joinpath(data_dir, "ncep_ncar")

	#### READ IN GRIDDED DATA ####
	variables = ["air", "pres", "rhum", "slp", "pr_wtr", "uwnd", "vwnd"]
	exten = ".mon.mean.nc"

	f1 = joinpath(ncep_ncar_path, "$(variables[1])$exten")

	hours_past_1800 = ncread(f1, "time")
	days_past_1800 = hours_past_1800 ./ 24
	dates_x = map(x -> init_date + Dates.Day(x), days_past_1800)
	info = ncinfo(f1)

	grid_lon_vals = ncread(f1, "lon")
	grid_lat_vals = ncread(f1, "lat")
	glon = (grid_lon_vals .>= min_lon) & (grid_lon_vals .<= max_lon)
	glat = (grid_lat_vals .>= min_lat) & (grid_lat_vals .<= max_lat)

	dimlat = sum(glat)
	dimlon = sum(glon)
	dimtime = info.dim["time"].dimlen

	X = zeros(dimlat*dimlon, length(variables), dimtime)

	for i =1:length(variables)
	    f = joinpath(ncep_ncar_path, "$(variables[i])$exten")
	    vardata = ncread(f, variables[i])[glon, glat, :]
	    vardata = reshape( reshape(vardata, 1, length(vardata))
	    		,size(vardata)[1] * size(vardata)[2], size(vardata)[3])
	    X[:, i, :] = vardata
	end

	println("Gridded Data\n", "Number of longitude: ", dimlon, "\nNumber of Latitude: ", dimlat, "\nNumber of variables: ", length(variables))

	##### READ IN OBSERVRED DATA FOR Y ####

	X = reshape(X, size(X)[1] * size(X)[2], size(X)[3])'
	info = ncinfo(gridded_data_file)

	lon_vals = (ncread(gridded_data_file, "longitude") + 360) % 360
	lat_vals = ncread(gridded_data_file, "latitude")
	lon = (lon_vals .>= min_lon) & (lon_vals .<= max_lon)
	lat = (lat_vals .>= min_lat) & (lat_vals .<= max_lat)

	dimlat = sum(lat)
	dimlon = sum(lon)
	dimtime = info.dim["time"].dimlen

	Y = zeros(dimlon, dimlat, dimtime, 1)
  Y[:, :, :, 1] = ncread(gridded_data_file, "Prcp")[lon, lat, :]

	rows = (dates_x .>= Date(min_year)) & (dates_x .< Date(max_year+1))
	println("Observed Data\nNumber of longitude: ", dimlon, "\nNumber of Latitude: ", dimlat, "\nNumber of variables: ", length(variables))
	out = ["X" => X[rows, :], "Y"=> Y, "obs_lon"=> lon_vals[lon], "obs_lat"=> lat_vals[lat], "grid_lat"=> grid_lat_vals[glat], "grid_lon"=> grid_lon_vals[glon]]
  return(out)
end
