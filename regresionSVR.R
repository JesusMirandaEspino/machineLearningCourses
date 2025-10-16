dataset = read.csv('C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/Position_Salaries.csv')

dataset = dataset[,2:3]


#install.packages("e1071")

#set.seed(123)
#split = sample.split( dataset$Profit, SplitRatio = 0.8 )
#training_set = subset( dataset, split == TRUE  )
#testing_set = subset( dataset, split == FALSE  )

#regression = lm(  formula = Profit ~ ., data = training_set)

#y_predict = predict( regression, newdata =  testing_set )


regression = svm( formula = Salary ~ ., data = dataset, type = "eps-regression", kernel = "radial" )




ggplot() +
  geom_point(aes( x = dataset$Level, y = dataset$Salary), 
             color="red") +
  geom_line(aes(  x = dataset$Level, 
                  y = predict(regression, newdata =  dataset)),
            color="blue") +
  ggtitle("Sueldo por nivel") +
  xlab("Nivel") +
  ylab("Salario")


y_pred = predict( regression,  newdata =  data.frame(Level = 6.5) )


