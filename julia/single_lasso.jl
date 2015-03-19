include("ElasticNetSVM.jl")
include("readncfiles.jl")

max_lat = 75
min_lat = 0
min_lon = 180
max_lon = 315
test_percentage = 0.33

ncdata = readdata(min_lat, max_lat, min_lon, max_lon)
X = ncdata["X"]
Y = ncdata["Y"]
obs_lat = ncdata["obs_lat"]
obs_lon = ncdata["obs_lon"]
grid_lon = ncdata["grid_lon"]
grid_lat = ncdata["grid_lat"]

X_train, _, X_test, _ = split_training(X, Y, test_percentage)
X_train, X_test = normalize_features(X_train, X_test)

test_lon = 360 - 122
test_lat = 47



test_lonidx = findfirst(obs_lon .> test_lon)
test_latidx = findfirst(obs_lat .> test_lat)

obs_lat[test_latidx]
obs_lon[test_lonidx]

y = reshape(Y[test_lonidx, test_latidx, :, :], size(Y)[3]*size(Y)[4])
_, y_train, _, y_test = split_training(X, y, test_percentage)

lambda=linspace(0.01, 0.5, 100)
cv = glmnetcv(X_train, y_train, alpha=1, lambda=lambda, nfolds=10) #, lambda_min_ratio=0.0001,
yhat_lasso_test = elastic_net_predict(cv, X_test)

i = indmin(cv.meanloss)
sum(cv.path.betas[:,i] .!= 0)
yhat = GLMNet.predict(cv.path, X_train)[:, i]
cor(yhat_lasso_test, y_test)
sum((yhat_lasso_test - y_test).^2)
cv.path.betas

x_plot = reshape(X[1,:], length(grid_lon), length(grid_lat), 7)
imagesc(x_plot[:, :, 7]')

cv.path.betas
b = cv.path.betas[:, indmin(cv.meanloss)]
beta_plot = reshape(b, length(grid_lon), length(grid_lat), 7)

grid_lat
grid_lat_id = findfirst(grid_lat .> test_lat)
grid_lon_id = findfirst(grid_lon .> test_lon)

imagesc(sum(beta_plot, 3)[:,:,1]' .> 0)

