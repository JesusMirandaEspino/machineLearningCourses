#install.packages("arules")
#install.packages("arulesViz")  
library(arules)
library(arulesViz)
dataset = read.csv('C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/Market_Basket_Optimisation.csv', header = FALSE)
dataset = read.transactions('C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/Market_Basket_Optimisation.csv', sep = ',', rm.duplicates = TRUE)


summary(dataset)
itemFrequencyPlot(dataset, topN = 10)


rules = apriori(data = dataset, parameter = list(support = 0.004, confidence = 0.2))


inspect(sort(rules, by = 'lift')[1:10])


plot(rules, method = "graph", engine = "htmlwidget")