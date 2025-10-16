dataset = read.csv('C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/Position_Salaries.csv')

dataset = dataset[,2:3]

#install.packages("caTools")

#set.seed(123)
#split = sample.split( dataset$Profit, SplitRatio = 0.8 )
#training_set = subset( dataset, split == TRUE  )
#testing_set = subset( dataset, split == FALSE  )

#regression = lm(  formula = Profit ~ ., data = training_set)

#y_predict = predict( regression, newdata =  testing_set )


lin_reg = lm(formula = Salary ~ ., data = dataset )



dataset$Level2 = dataset$Level^2
dataset$Level3 = dataset$Level^3  
dataset$Level4 = dataset$Level^4 

poly_reg = lm(formula = Salary ~ ., data = dataset )


x_grid = sep( min(dataset$Level), max(dataset$Level), 0.1 )

ggplot() +
  geom_point(aes( x = dataset$Level, y = dataset$Salary), 
             color="red") +
  geom_line(aes(  x = dataset$Level, 
                  y = predict(lin_reg, newdata =  dataset)),
            color="blue") +
  ggtitle("Sueldo por nivel") +
  xlab("Nivel") +
  ylab("Salario")



ggplot() +
  geom_point(aes( x = dataset$Level, y = dataset$Salary), 
             color="red") +
  geom_line(aes(  x = dataset$Level, 
                  y = predict(poly_reg, newdata =  dataset)),
            color="blue") +
  ggtitle("Sueldo por nivel") +
  xlab("Nivel") +
  ylab("Salario")


y_pred = predict( lin_reg,  newdata =  data.frame(Level = 6.5) )

y_pred_poly = predict( poly_reg,  newdata =  data.frame(Level = 6.5, 
                                                  Level2 = 6.5^2,
                                                  Level3 = 6.5^3,
                                                  Level4 = 6.5^4) )


