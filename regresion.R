dataset = read.csv('C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/Salary_Data.csv')



#install.packages("caTools")

set.seed(123)
split = sample.split( dataset$Salary, SplitRatio = 0.8 )
training_set = subset( dataset, split == TRUE  )
testing_set = subset( dataset, split == FALSE  )


#training_set[,2:3] = scale( training_set[,2:3] )
#testing_set[,2:3] = scale( testing_set[,2:3] )

regression = lm(  formula = Salary ~ YearsExperience, data = training_set)

y_predict = predict( regression, newdata =  testing_set )
#install.packages("ggplot2")



ggplot() +
  geom_point(aes( x = training_set$YearsExperience, 
                  y = training_set$Salary), 
            color="red") +
  geom_line(aes(  x = training_set$YearsExperience, 
                  y = predict(regression, newdata =  training_set)),
            color="blue") +
  ggtitle("Sueldo por años de experiencia") +
  xlab("Años de experiencia") +
  ylab("Salario")


ggplot() +
  geom_point(aes( x = testing_set$YearsExperience, 
                  y = testing_set$Salary), 
             color="red") +
  geom_line(aes(  x = training_set$YearsExperience, 
                  y = predict(regression, newdata =  training_set)),
            color="blue") +
  ggtitle("Sueldo por años de experiencia Testing") +
  xlab("Años de experiencia") +
  ylab("Salario")

