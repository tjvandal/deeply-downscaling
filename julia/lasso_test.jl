using GLMNet
n = 100000
m = 1000
x = randn(n,m)
y = x[:,1] + x[:,2] + randn(n).*1.5

path = glmnet(x, y, alpha=1, lambda=[0.5])
yhat = GLMNet.predict(path, x)
sse = sum((yhat - y).^2)
cor(yhat, y)

ll = linspace(0.1,1,10)
cv = glmnetcv(x, y, lambda=ll, nfolds=5)
