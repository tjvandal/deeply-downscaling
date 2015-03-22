## next steps
# 1. moving window
# 2. shrink size of window
# elif interpolation

include("readncfiles.jl")
@everywhere using GLMNet
#@everywhere using Winston
@everywhere using PyCall
@everywhere using DataFrames
#@everywhere using MATLAB
@everywhere using MultivariateStats

function split_training(X, y, test_percent=0.33)
	test_idx = round(size(X)[1] * (1-test_percent)) - 2
	return(X[1:test_idx,:], y[1:test_idx], X[(test_idx+1):end, :], y[(test_idx+1):end])
end

function normalize_features(X_train, X_test)
	mu = mean(X_train, 1)
	sd = std(X_train, 1)
	X_train = (X_train .- mu) ./ sd
	X_test = (X_test .- mu) ./ sd
	return(X_train, X_test)
end

@everywhere function svr_train(xtrain,ytrain)
  @pyimport sklearn.svm as svm
  @pyimport sklearn.grid_search as grid_search

  parameters = {"C"=>Float64[10^x for x in linspace(-2, 2, 1)],
    "epsilon"=>Float64[10^x for x in linspace(-2, 2, 2)]}
  svr = svm.SVR(kernel="linear")
  clf = grid_search.GridSearchCV(svr, parameters, cv=2)
  clf[:fit](xtrain, ytrain)
  svr_model = clf["best_estimator_"]
  return(svr_model)
end

@everywhere function elastic_net_predict(model, x)
    min_mse_idx = indmin(model.meanloss)
    yhat = GLMNet.predict(model.path, x)[:, min_mse_idx]
    return(yhat)
end

@everywhere function elastic_net_nonzero_variables(cv, x)
  min_mse_idx = indmin(cv.meanloss)
  nonzero_cols = cv.path.betas[:,min_mse_idx] .!= 0
  return(x[:, nonzero_cols])
end

@everywhere function save_data(xtrain, ytrain, xtest, ytest)
  writecsv("x_train.csv", X_train)
  writecsv("x_test.csv", X_test)
  writecsv("y_train.csv", y_train)
  writecsv("y_test.csv", y_test)
  return(true)
end

## function is not being used, replaced with regular svr via pycall
@everywhere function lssvm_train(xtrain, ytrain, xtest)
  sess = MSession(rand(1:100000))
  mx_train = mxarray(xtrain)
  my_train = mxarray(ytrain)
  mx_test = mxarray(xtest)
  put_variable(sess, :mx_train, mx_train)
  put_variable(sess, :my_train, my_train)
  put_variable(sess, :mx_test, mx_test)

  eval_string(sess, "addpath '/home/tj/Desktop/MachineLearning/Project/StatLSSVM';")
  eval_string(sess, "model=initlssvm(mx_train, my_train, [], [], 'RBF_kernel');")
  eval_string(sess, "model = tunelssvm(model, 'crossval', {10, 'mae'});")
  eval_string(sess, "model_train = trainlssvm(model);")
  eval_string(sess, "my_hat_test = simlssvm(model_train, mx_test);")

  my_hat_test = get_mvariable(sess, :my_hat_test)
  close(sess)
  return(jarray(my_hat_test))
end

@everywhere function whiten_pca(xtrain, xtest)
  cov_x_train = cov(xtrain)
  D, W = eig(cov_x_train)
  I = sortperm(D, rev=true)
  W_sortDesc = W[:, I]
  W_X_train = xtrain * W_sortDesc
  W_X_test = xtest * W_sortDesc
  M = MultivariateStats.fit(PCA, xtrain)
  explained = cumsum(MultivariateStats.principalvars(M)/MultivariateStats.tprincipalvar(M))
  count = sum(explained .< 0.85)
  W_X_train = W_X_train[:, 1:count]
  W_X_test = W_X_test[:, 1:count]
  return(W_X_train, W_X_test)
end

@everywhere function mse(x1, x2)
  return(mean((x1-x2).^2))
end


@everywhere function main(X_test, X_train, y_train, y_test, lat_index, lon_index, alpha=0.5, x_train_pca=None, x_test_pca=None)
  srand(3)

  ## Train Lasso
  cv = glmnetcv(X_train, y_train, alpha=1, lambda_min_ratio=0.0001, nfolds=10, nlambda=1000)
  yhat_lasso_test = elastic_net_predict(cv, X_test)

  ## Train Elastic Net
  cv = glmnetcv(X_train, y_train, alpha=alpha, lambda_min_ratio=0.0001, nfolds=10, nlambda=1000)
  yhat_el_test = elastic_net_predict(cv, X_test)
  elnet_corr = cor(yhat_el_test, y_test)
#  println("correlation for lasso : ", elnet_corr, " Mean of yhat_test: ", mean(yhat_el_test), " Variance of yhat_test: ", var(yhat_el_test))

  X_train_el = elastic_net_nonzero_variables(cv, X_train)
  X_test_el = elastic_net_nonzero_variables(cv, X_test)
  #println("Number of Non Zero Betas: ", size(X_train_el)[2])

  ## Train SVR
  lassosvr_model = svr_train(X_train_el, y_train)
  y_hat_svr_test = lassosvr_model[:predict](X_test_el)

  lasso_svr_corr = cor(y_hat_svr_test, y_test)
  #println("Correlation ElasticNet-SVR:", lasso_svr_corr)

  ## Train LS-SVM
  #yhat_el_lssvm_test = lssvm_train(X_train_el, y_train, X_test_el)
  #elnet_lssvm_cor = cor(yhat_el_lssvm_test, y_test)

  ## Train PCA-SVM
  if (typeof(x_train_pca) == Bool) & (typeof(x_test_pca) == Bool)
    x_train_pca, x_test_pca = whiten_pca(X_train, X_test)
  end
  #yhat_pca = lssvm_train(x_train_pca, y_train, x_test_pca)
  pcasvr_model = svr_train(x_train_pca, y_train)
  yhat_pca = pcasvr_model[:predict](x_test_pca)

  results = {"test_mean"=>[mean(y_test)], "test_std"=>[std(y_test)],
                 "lasso_mse"=>[mse(y_test, yhat_lasso_test)], "lasso_mean"=>[mean(yhat_lasso_test)], "lasso_std"=>[std(yhat_lasso_test)],"lasso_cor"=>[cor(yhat_lasso_test, y_test)[1]],
                 "el_mse"=>[mse(y_test, yhat_el_test)], "el_mean"=>[mean(yhat_el_test)], "el_std"=>[std(yhat_el_test)], "el_cor"=>[cor(yhat_el_test, y_test)[1]],
                 "el-svr_mse"=>[mse(y_test, y_hat_svr_test)], "el-svr_mean"=>[mean(y_hat_svr_test)], "el-svr_std"=>[std(y_hat_svr_test)], "el-svr_cor"=>[cor(y_hat_svr_test, y_test)[1]],
                 "pca-svr_mse"=>[mse(y_test, yhat_pca)], "pca-svr_mean"=>[mean(yhat_pca)], "pca-svr_std"=>[std(yhat_pca)], "pca-svr_cor"=>[cor(yhat_pca, y_test)[1]]
            }

  return(convert(DataFrame,results))
end

function run_all()
  max_lat = 75
  min_lat = 0
  min_lon = 180
  max_lon = 315
  test_percentage = 0.33

  ncdata = readdata(min_lat, max_lat, min_lon, max_lon)  ##readncfiles.jl
  X = ncdata["X"]
  Y = ncdata["Y"]
  obs_lat = ncdata["obs_lat"]
  obs_lon = ncdata["obs_lon"]
  grid_lon = ncdata["grid_lon"]
  grid_lat = ncdata["grid_lat"]

  X_train, _, X_test, _ = split_training(X, Y, test_percentage)
  X_train, X_test = normalize_features(X_train, X_test)

  pca_train_file = "pca_train_x"
  pca_test_file = "pca_test_x"
  if isfile(pca_train_file) & isfile(pca_test_file)
      x_train_pca = readdlm(pca_train_file)
      x_test_pca = readdlm(pca_test_file)
  else
      println("Running PCA Transform")
      x_train_pca, x_test_pca = whiten_pca(X_train, X_test)
      writedlm(pca_train_file, x_train_pca)
      writedlm(pca_test_file, x_test_pca)
  end



  df = false
  tasks=Array(RemoteRef, size(Y)[2], size(Y)[1])
  lon_skip = round(size(Y)[1]/10)
  lat_skip = round(size(Y)[2]/10)

  test_lonidx = 200
  test_latidx = 111

  for lonidx=1:size(Y)[1]
#    if lonidx % lon_skip != 0
#      continue
#    end
    println("$(lonidx*100./size(Y)[1]) percent completed")
    for latidx=1:size(Y)[2]
#      if latidx % lat_skip != 0
#        continue
#      end

      y = reshape(Y[lonidx, latidx, :, :], size(Y)[3]*size(Y)[4])
      _, y_train, _, y_test = split_training(X, y, test_percentage)
      if mean(y_train) < 10000
        tasks[latidx, lonidx] = @spawn main(X_test, X_train, y_train, y_test, latidx, lonidx, 0.5, x_train_pca, x_test_pca)
      end

    for latidx=1:size(Y)[2]
#      if latidx % lat_skip != 0
#        continue
#      end

      if !isdefined(tasks, latidx, lonidx)
          continue
      end
      res = deepcopy(fetch(tasks[latidx, lonidx]))
      if typeof(res) == Bool
        continue
      else

        res[:latitude] = obs_lat[latidx]
        res[:longitude] = obs_lon[lonidx]
        if (typeof(df) != Bool)
          append!(df, res)
        else
          df = res
        end
      end
    end
    if (typeof(df) != Bool)
      println(df)
      writetable("results.csv", df)
    end
  end
  end
end
