########################################################################################################

############################################ CHURN REDUCTION ###########################################

########################################################################################################


#remove all the objects stored
rm(list=ls())


#Current working directory
getwd()

setwd("D:/project1/churnreduction")
##################################  Loading Data Test.csv and train.csv##################################################

train_dataset = read.csv("Train_data.csv",header = TRUE)
train_dataset$myindex=1
test_dataset=read.csv("Test_data.csv",header = TRUE)
test_dataset$myindex=2
mydata=rbind(train_dataset,test_dataset)
rm(test_dataset,train_dataset)

##Getting the column names of the dataset
colnames(mydata)

###Getting the structure of the dataset
str(mydata)

##Getting the number of variables and obervation in the datasets
dim(mydata)

###################################### MISSING VALUE ANALYSIS ########################################################

sapply(mydata,function(x){sum(is.na(x))})
sapply(mydata,function(x){sum(is.null(x))})

####################################  DATA MANIPPULATION #############################################################

##Areacode is given numeric type so we are converting it into factor type 
mydata$area.code=as.factor(mydata$area.code)
mydata$myindex=as.factor(mydata$myindex)
str(mydata)

##Data Manupulation; convert string categories into factor numeric
for(i in 1:ncol(mydata)){
  
  if(class(mydata[,i]) == 'factor'){
    
    mydata[,i] = factor(mydata[,i], labels=(1:length(levels(factor(mydata[,i])))))
    
  }
}


##################################### OUTLIER ANALYSIS #############################################################

numeric_index = sapply(mydata,is.numeric) #selecting only numeric
numeric_data = mydata[,numeric_index]
cnames = colnames(numeric_data)
View(cnames)

library(ggplot2)
for (i in 1:length(cnames))
{
  assign(paste0("gn",i), ggplot(aes_string(y = (cnames[i]), x = "Churn"), data = mydata)+ 
           stat_boxplot(geom = "errorbar", width = 0.5) +
           geom_boxplot(outlier.colour="red", fill = "grey" ,outlier.shape=18,
                        outlier.size=1, notch=FALSE) +#            theme(legend.position="bottom")+
           labs(y=cnames[i],x="Churn")+
           ggtitle(paste("Box plot of Churn for",cnames[i])))
}

# ## Plotting plots together
gridExtra::grid.arrange(gn1,gn2,gn3,ncol=3)
gridExtra::grid.arrange(gn4,gn5,gn6,ncol=3)
gridExtra::grid.arrange(gn7,gn8,gn9,ncol=3)
gridExtra::grid.arrange(gn10,gn11,gn12,ncol=3)
gridExtra::grid.arrange(gn13,gn14,gn15,ncol=3)


#Replace all outliers with NA and imputing them using KNNimutation
#create NA on outliers
for(i in cnames){
  val = mydata[,i][mydata[,i] %in% boxplot.stats(mydata[,i])$out]
  print(length(val))
  mydata[,i][mydata[,i] %in% val] = NA
}
sum(is.na(mydata))

##KNNimputation
library(DMwR)
mydata = knnImputation(mydata, k = 5)


########################################## FEATURE SELECTION ########################################################

library(corrgram)
corrgram(mydata[,numeric_index], order = F,
         upper.panel=panel.pie, text.panel=panel.txt, main = "Correlation Plot")

## Chi-squared Test of Independence
library(MASS)
factor_index = sapply(mydata,is.factor)
factor_data = mydata[,factor_index]
View(factor_data)
for (i in 1:5)
{
  print(names(factor_data)[i])
  print(chisq.test(table(factor_data$Churn,factor_data[,i]),simulate.p.value = TRUE))
}

colnames(mydata)


##Dimension reduction
reduced_mydata = subset(mydata,select=-c(area.code,phone.number,total.day.minutes,
                                         total.eve.minutes,total.night.minutes,total.intl.minutes))
dim(reduced_mydata)
str(reduced_mydata)


#########################################  FEATURE SCALING ############################################################

qqnorm(reduced_mydata$total.night.calls)
hist(reduced_mydata$account.length)

##Standardisation
reduced_numeric_index = sapply(reduced_mydata,is.numeric) #selecting only numeric
reduced_numeric_data = reduced_mydata[,reduced_numeric_index]
reduced_cnames = colnames(reduced_numeric_data)
View(reduced_cnames)
 for(i in reduced_cnames)
  {
   print(i)
   reduced_mydata[,i] = (reduced_mydata[,i] - mean(reduced_mydata[,i]))/
                                  sd(reduced_mydata[,i])
 }

#############################################  SAMPLING ###############################################################

train=reduced_mydata[reduced_mydata$myindex==1,]
test=reduced_mydata[reduced_mydata$myindex==2,]
train$myindex=NULL
test$myindex=NULL


############################################## MODELING BUILDING ######################################################

################################## DECISION TREEE FRO CLASSIFICATION ##################################################
library(C50)
library(caret)
C50_model = C5.0(Churn ~., train, trials =100, rules = TRUE)

#Summary of DT model
summary(C50_model)

#Lets predict for test cases
C50_Predictions = predict(C50_model, test[,-15], type = "class")

##Evaluate the performance of classification model
ConfMatrix_C50 = table(test$Churn, C50_Predictions)
confusionMatrix(ConfMatrix_C50)

#False Negative rate
#FNR = FN/FN+TP 
##accuracy : 94.06%
##FNR :43.30%

################################################# RANDOM FOREST #####################################################

library(randomForest)
library(RRF)
library(inTrees)
RF_model = randomForest(Churn ~ ., train, importance = TRUE, ntree = 100)

#Extract rules fromn random forest
#transform rf object to an inTrees' format
 treeList = RF2List(RF_model)  
 
# #Extract rules
 exec = extractRules(treeList, train[,-15])  # R-executable conditions
# 
# #Visualize some rules
 exec[1:2,]
 
# #Make rules more readable:
 readableRules = presentRules(exec, colnames(train))
 readableRules[1:2,]
 
# #Get rule metrics
 ruleMetric = getRuleMetric(exec, train[,-15], train$Churn)  # get rule metrics
 
# #evaulate few rules
 ruleMetric[1:2,]

#Presdict test data using random forest model
RF_Predictions = predict(RF_model, test[,-15])

##Evaluate the performance of classification model
ConfMatrix_RF = table(test$Churn, RF_Predictions)
confusionMatrix(ConfMatrix_RF)

#False Negative rate
#FNR = FN/FN+TP 

#Accuracy = 92.08 %
#FNR = 52.67%

########################################## LOGISTIC REGRESSION ######################################################


logit_model = glm(Churn ~ ., data = train, family = "binomial")

#summary of the model
summary(logit_model)

#predict using logistic regression
logit_Predictions = predict(logit_model, newdata = test, type = "response")

#convert prob
logit_Predictions = ifelse(logit_Predictions > 0.5, 1, 0)


##Evaluate the performance of classification model
ConfMatrix_LR = table(test$Churn, logit_Predictions)
View(ConfMatrix_LR)

#Accuracy: 87.43 %
#FNR:80.80 %


################################# KNN IMPLEMENTATION ##############################################################

library(class)

#Predict test data
KNN_Predictions = knn(train[, 1:14], test[, 1:14], train$Churn, k = 7)

#Confusion matrix
Conf_matrix = table(KNN_Predictions, test$Churn)
View(Conf_matrix)
#Accuracy
sum(diag(Conf_matrix))/nrow(test)

#Accuracy: 86.98 %
#FNR: 33.33 %

############################################ NAIVE BAYES ##############################################################

library(e1071)

#Develop model
NB_model = naiveBayes(Churn ~ ., data = train)

#predict on test cases #raw
NB_Predictions = predict(NB_model, test[,1:14], type = 'class')

#Look at confusion matrix
Confmatrix_NB = table(observed = test[,15], predicted = NB_Predictions)
confusionMatrix(Confmatrix_NB)

#Accuracy: 87.76 %
#FNR: 79.01 %
