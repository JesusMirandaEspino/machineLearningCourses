dataset = read.csv('C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/50_Startups.csv')


dataset$State = factor( dataset$State, 
                          levels = c("New York", "California", "Florida"), 
                          labels = c(1,2,3) )


#install.packages("caTools")

set.seed(123)
split = sample.split( dataset$Profit, SplitRatio = 0.8 )
training_set = subset( dataset, split == TRUE  )
testing_set = subset( dataset, split == FALSE  )

regression = lm(  formula = Profit ~ ., data = training_set)

y_predict = predict( regression, newdata =  testing_set )


SL = 0.05
regression = lm(  formula = Profit ~ R.D.Spend + Administration +  
                    Marketing.Spend + State, data = dataset)

summary(regression)



regression = lm(  formula = Profit ~ R.D.Spend + Administration +  
                    Marketing.Spend, data = dataset)
summary(regression)

regression = lm(  formula = Profit ~ R.D.Spend + Administration +  
                    Marketing.Spend, data = dataset)
summary(regression)


regression = lm(  formula = Profit ~ R.D.Spend +  
                    Marketing.Spend, data = dataset)
summary(regression)


regression = lm(  formula = Profit ~ R.D.Spend, data = dataset)
summary(regression)


#install.packages("https://cran.r-project.org/src/contrib/Archive/ElemStatLearn/ElemStatLearn_2015.6.26.2.tar.gz",repos=NULL, type="source")



