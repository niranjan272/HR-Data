library(corrplot)
library(caret)
library(rpart)
library(C50)
library(pROC)
library(randomForest)
set.seed(43)
setwd("E:\\Coursera\\HR Data\\")
hr_data <- read.csv("E:\\Coursera\\HR Data\\HR_comma_sep.csv")
print(head(hr_data),3)
ncol(hr_data)
#To find a correlationship between all the variables
correlation_hr = cor(hr_data[,c(1:8)])
print(correlation_hr)
corrplot(correlation_hr,method = "circle")


#Split Data
nrow(hr_data)
indexes = sample(1:nrow(hr_data),size = 0.3*nrow(hr_data))
test = hr_data[indexes,]
train = hr_data[-indexes,]
nrow(test)
nrow(train)

#classification algorithm
hr_data_left <- rpart(as.factor(left)~.,data = train,method = "class")
printcp(hr_data_left)
plotcp(hr_data_left)
summary(hr_data_left)
plot(hr_data_left,uniform = TRUE)

hr_data_left_predict <- predict(hr_data_left,train)
head(hr_data_left_predict)
confusionMatrix(hr_data_left_predict,test$left)


ctrl <- trainControl(method = "cv", number = 5)
tbmodel <- train(as.factor(left) ~., data = train, method = "C5.0Tree", trControl = ctrl)
pred  <- predict(tbmodel,test)
confusionMatrix(pred,test$left)
roc <- roc(as.numeric(test$left),as.numeric(pred))
plot(roc, ylim=c(0,1), print.thres=TRUE, main=paste('AUC:',round(roc$auc[[1]],3)),col = 'blue')


#Random Forest
colnames(hr_data)
hr_data_random_forest_train <- randomForest(train[,c(1:6,8:10)],as.factor(train[,7]),ntree = 50,do.trace = T)
summary(hr_data_random_forest_train)
pred_random_forest <- predict(hr_data_random_forest_train,test[,c(1:6,8:10)])
pred_random_forest
confusionMatrix(data = pred_random_forest,test$left)

hr_data_random_forest_train$confusion[,'class.error']
conf <- hr_data_random_forest_train$confusion
conf
#Check for relationship between different variable
employee_regression <- lm(satisfaction_level ~ number_project + average_montly_hours + promotion_last_5years
                          ,data = hr_data)
summary(employee_regression)

employee_regression_salary <- glm(satisfaction_level ~ salary + number_project+ average_montly_hours ,
                                  data = hr_data)
summary(employee_regression_salary)
plot(hr_data$salary,hr_data$satisfaction_level)

satisfaction_level_left = lm(last_evaluation ~ satisfaction_level,data = hr_data)
summary(satisfaction_level_left)
plot(hr_data$satisfaction_level,hr_data$last_evaluation)
abline(satisfaction_level_left)

colnames(hr_data)
relationship_project_accident  <- lm(salary ~ number_project + Work_accident,data = hr_data)
summary(relationship_project_accident)




