train<-read.csv("eBayiPadTrain.csv", encoding="UTF-8")
test<-read.csv("eBayiPadTest.csv", encoding="UTF-8")

#Dealing with text data
library(tm)
library(SnowballC)
library(stringr)
train$description<-str_replace_all(train$description,"[^[:graph:]]", " ") 
corpus<-Corpus(VectorSource(usableText))
corpus<-tm_map(corpus,tolower)
corpus<-tm_map(corpus,removePunctuation)
corpus<-tm_map(corpus,removeWords,c("apple","ipad",stopwords("en")))
corpus<-tm_map(corpus,stemDocument)
frequencies<-DocumentTermMatrix(corpus)
sparse<-removeSparseTerms(frequencies,0.99)
ebaySparse<-as.data.frame(as.matrix(sparse))
colnames(ebaySparse)<-make.names(colnames(ebaySparse))

#Combining text analysis to main dataset
library(janitor)
train$description<-NULL
train$UniqueID<-NULL
trdata<-cbind(ebaySparse,train)
trdata$sold<-as.factor(trdata$sold)
clean_names(trdata)
colSums(is.na(trdata))


#Building predictive models
#CART Model
library(rpart.plot)
set.seed(1)
ebayCART<-rpart(sold~.,data=trdata,method="class",minbucket=50)
prp(ebayCART)
predictCART<-predict(ebayCART,newdata=trdata,type="class")
#Accuracy of CART Model
table<-table(trdata$sold,predictCART)
CARTacc<-(table[1,1]+table[2,2])/(table[1,1]+table[2,2]+table[1,2]+table[2,1])
CARTacc

#Improving CART through CV
library(e1071)
library(caret)
numFolds<-trainControl(method="cv",number=10)
cpGrid<-expand.grid(cp=seq(0.01,0.5,0.01))
train(sold~.,data=trdata,method="rpart",trControl=numFolds,tuneGrid=cpGrid)
ebayCV<-rpart(sold~.,data=trdata,method="class",cp=0.01)
predictCV<-predict(ebayCV,newdata=trdata,type="class")
table<-table(trdata$sold,predictCV)
CVacc<-(table[1,1]+table[2,2])/(table[1,1]+table[2,2]+table[1,2]+table[2,1])
CVacc
                                            
#RandomForest
library(randomForest)
ebayForest<-randomForest(sold~.,data=trdata,nodesize=25,ntree=400)
plot(ebayForest)
predictForest<-predict(ebayForest,trdata)
table<-table(trdata$sold,predictForest)
Forestacc<-(table[1,1]+table[2,2])/(table[1,1]+table[2,2]+table[1,2]+table[2,1])
Forestacc

#Logistic Regression
ebayLog<-glm(sold~.,data=trdata,family=binomial)
plot(ebayLog)
predictLog<-predict(ebayLog,trdata,type="response")
table<-table(train$sold,predictLog>0.5)
Logacc<-(table[1,1]+table[2,2])/(table[1,1]+table[2,2]+table[1,2]+table[2,1])
Logacc

#Improve with ROC
library(ROCR)
ROCRpred<-prediction(predictLog,trdata$sold)
ROCRperf<-performance(ROCRpred,"tpr","fpr")
plot(ROCRperf)
table<-table(train$sold,predictLog>0.7)
ROCacc<-(table[1,1]+table[2,2])/(table[1,1]+table[2,2]+table[1,2]+table[2,1])
ROCacc

# Our best model for predicting iPad sales is thus our random forest model with an Accuracy of 85%!
View(ebayForest)


