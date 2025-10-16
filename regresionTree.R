dataset = read.csv('C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/Position_Salaries.csv')

dataset = dataset[,2:3]



regression = rpart( formula = Salary ~ ., data = dataset, control=rpart.control(minsplit=1) )


#install.packages("rpart")

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