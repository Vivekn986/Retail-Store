#importing packages

library(tidymodels)
library(visdat)
library(tidyr)
library(car)
library(pROC)
library(ggplot2)
library(tidyr)
library(ROCit)
library(stringr)
library(dbplyr)


#this is retail store project . we have two data set  train and test sets. test data contains all data except store.

#loading the data(train and test data)
s_train=read.csv(r"(C:\DATASETS\store_train.csv)",stringsAsFactors = FALSE)
s_test=read.csv(r"(C:\DATASETS\store_test.csv)",stringsAsFactors = FALSE)




#Understanding Our Data

#Each row represents characteristic of a single planned store.We can see from above that many categorical data has been coded to mask the data.Here is the interpretation for the columns Id : store id numeric sale figures for 5 types : sales0 sales1 sales2 sales3 sales4

#country : categorical :: coded values for country

#State : categorical :: coded values for State

#CouSub : numeric ::subscription values at county level

#countyname : Categorical ::county names

#storecode : categorical :: store codes

#Areaname : categorical :: name of the area , many times it matches with county name

#countytownname : categorical :: county town name

#population : numeric :: population of the store area

#state_alpha : categorical :: short codes for state

#store_Type : categorical :: type of store

#store : categorical 1/0 : target indicator var 1=opened 0=not opened



#info of train and test data
glimpse(s_train)
glimpse(s_test)
head(s_test)
head(s_train)
setdiff(names(s_train),names(s_test))

#checking na values in train columns
lapply(s_train,function(x)sum(is.na(x)))

#combining two datasets.we'll add store column to test bcs columns need to be same for two datasets..We are also going to add an identifier column ‘data. which will recognize whether it is from train or test.

s_test$store = NA
s_train$data='train'
s_test$data='test'
s_all=rbind(s_train,s_test)
s_all=s_all %>% 
  mutate(s1=substr(storecode,1,5)) %>% 
  select(-storecode)

#seperating the data sets.
s_train=s_all %>% filter(data=='train') %>% select(-data)
s_test=s_all %>% filter(data=='test') %>% select(-data,-store)

#we using random forest so we going to convert store data into factor type using as.factor. This is how random forest differentiate from regression and classification.
s_train$store = as.factor(s_train$store)

#Data Preparation and creating dummies

dp_pipe=recipe(store~.,data=s_train) %>% 
  update_role(countytownname,new_role = "drop_vars") %>%
  update_role(state_alpha,s1,store_Type,countyname,Areaname,new_role="to_dummies") %>% 
  step_rm(has_role("drop_vars")) %>% 
  step_unknown(has_role("to_dummies"),new_level="__missing__") %>% 
  step_other(has_role("to_dummies"),threshold =0.02,other="__other__") %>% 
  step_dummy(has_role("to_dummies")) %>%
  step_impute_median(all_numeric(), -all_outcomes())

dp_pipe=prep(dp_pipe)

train=bake(dp_pipe,new_data=NULL)
test=bake(dp_pipe,new_data=s_test)



#Next we will break our train data into 2 parts in 80:20 ratio. We will build model on one part & check its performance on the other.
set.seed(2)
dim(train)
s=sample(1:nrow(train),0.8*nrow(train))
t1=train[s,] ## you create as model
t2=train[-s,]

#Model Building
#load package random forest.
library(randomForest)

library(cvTools)


#Regression Model Building with Random Forest-
#As we know that RandomForest has 4 parameters- mtry,ntree,maxnodes & nodessize.Let us set the parameter values that we want to try out.

param=list(mtry=c(5,10,15,20,25,35),
           ntree=c(50,100,200,500,700),
           maxnodes=c(5,10,15,20,30,50,100),
           nodesize=c(1,2,5,10)
)

mycost_auc=function(y,yhat){
  roccurve=pROC::roc(y,yhat)
  score=pROC::auc(roccurve)
  return(score)
}

#it will create 5*5*6*4 = 600 possible combination.it will take more time to calculate for all.instead we will use randomly select smaller subset .

subset_paras=function(full_list_para,n=10){  #n=10 is default, you can give higher value
  
  all_comb=expand.grid(full_list_para)
  
  s=sample(1:nrow(all_comb),n)
  
  subset_para=all_comb[s,]
  
  return(subset_para)
}

num_trials=50
my_params=subset_paras(param,num_trials)

myauc=0
for(i in 1:num_trials){
  print(paste('starting iteration :',i))
  # uncomment the line above to keep track of progress
  params=my_params[i,]
  k=cvTuning(randomForest,store~.,
             data =train,
             tuning =params,
             folds = cvFolds(nrow(train), K=10, type ="random"),
             cost =mycost_auc, seed =2,
             predictArgs = list(type="prob")
  )
  score.this=k$cv[,2]
  if(score.this>myauc){
    print(params)
    # uncomment the line above to keep track of progress
    myauc=score.this
    print(myauc)
    
    # uncomment the line above to keep track of progress
    best_params=params
  }
  print('DONE')
  # uncomment the line above to keep track of progress
}
myauc
#this is best auc score = 0.8442616
best_params
#these is best_params
#mtry ntree maxnodes nodesize
#544   20    50       30        5

#Lets use these to Build our model.
ci.rf.final=randomForest(store~.-Id,
                         mtry= best_params$mtry,
                         ntree= best_params$ntree,
                         maxnodes= best_params$maxnodes,
                         nodesize=  best_params$nodesize,
                         data=train
)
ci.rf.final

test.score=predict(ci.rf.final,newdata = test,type='prob')[,2]


varImpPlot(ci.rf.final)