# Retail-Store



This is a retail project where our challenge is to predict whether a retail store should get opened or not based on certain factors such as sales, population,area etc. We have been given two datasets store_train.csv and store_test.csv .We need to use data store_train to build predictive model for response variable ‘store’. store_test data contains all other factors except ‘store’, we need to predict that using the model that we will develop. We will be submitting our predicted values in terms of probability scores. This is a typical classification problem & we will use random forest for model building.


If you are using decision trees or random forest here, probability scores can be calculated as

#Random Forest
score=predict(rf_model,newdata= testdata, type="prob")[,2]

#Tree Model
score=predict(tree_model,newdata= testdata, type=‘vector’)[,2]
