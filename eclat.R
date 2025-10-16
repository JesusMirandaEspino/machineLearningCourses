library(arules)
library(arulesViz)
dataset = read.csv('C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/Market_Basket_Optimisation.csv', header = FALSE)
dataset = read.transactions('C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/Market_Basket_Optimisation.csv', sep = ',', rm.duplicates = TRUE)

summary(dataset)
itemFrequencyPlot(dataset, topN = 10)

rules = eclat(data = dataset, parameter = list(support = 0.003, minlen = 2))


inspect(sort(rules, by = 'support')[1:10])