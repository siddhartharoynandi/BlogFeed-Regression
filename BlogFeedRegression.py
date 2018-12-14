import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

train = pd.read_csv("/Users/siddhartharoynandi/Desktop/BlogFeedback/blogData_train.csv", header=None)
n,d = train.shape
#print n,d -- 52397,281
Xtrain = train.iloc[:,0:280]
ytrain = train.iloc[:,-1]
regr = linear_model.LinearRegression()
regr.fit(Xtrain, ytrain)


test = pd.read_csv("/Users/siddhartharoynandi/Desktop/BlogFeedback/blogData_test-2012.03.31.01_00.csv", header=None)
Xtest = test.iloc[:,0:280]
ytest = test.iloc[:,-1]
pred_y = regr.predict(Xtest)
SSE = 0.0
for idx,i in enumerate(ytest):
    SSE = SSE + ((ytest[idx] - pred_y[idx]) ** 2)
f = open('/Users/siddhartharoynandi/Desktop/ADM/CIS563_HW3/Regression_Output.txt', 'w')
f.write('Total SSE: '+str(SSE))
f.close()
#print("Total SSE: %.2f"%SSE)
exit(0)
#print("Mean squared error: %.2f"% mean_squared_error(ytest, pred_y)) #1632.16 * 214