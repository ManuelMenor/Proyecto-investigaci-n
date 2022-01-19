install.packages('C50')
install.packages('gmodels')
install.packages('mlr')

# Importamos dataset
dataset = read.csv('dataset/car-evaluation/car.csv')

# Analizamos la distribucion por criterios
library(ggplot2)
# Creamos el hisograma de compras
ggplot(data=dataset,aes(x=buying)) + geom_bar() + ggtitle("Histogram buying") + facet_wrap(~car)
# Histograma de mantenimiento
ggplot(data=dataset,aes(x=maint)) + geom_bar() + ggtitle("Histogram maintenance") + facet_wrap(~car)
# Histograma de puertas
ggplot(data=dataset,aes(x=doors)) +geom_bar() + ggtitle("Histogram doors") + facet_wrap(~car)
# Histograma de personas
ggplot(data=dataset,aes(x=persons)) + geom_bar() + ggtitle("Histogram persons") + facet_wrap(~car)
# Histograma de el tamaño del maletero
ggplot(data=dataset,aes(x=lug_boots)) + geom_bar() + ggtitle("Histogram lugage boots") + facet_wrap(~car)
# Histograma de la seguridad
ggplot(data=dataset,aes(x=safety)) + geom_bar() + ggtitle("Histogram safety") + facet_wrap(~car)

# Los clasificamos 
dataset$car = factor(dataset$car, levels = c('unacc', 'acc', 'good', 'vgood'))

# Aplicamos randomized splitting data set en el training set and test set
library(caTools)
split = sample.split(dataset$car, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Aplicamos el algoritmo c50 en el training set
library(C50)
decTree = C5.0(formula = car ~ ., data = training_set)
decTree

# Usamos el test set para predecir la precision de la clasificacion 
y_pred = predict(decTree, newdata = test_set[-7], type = 'class')

# Creamos arbol
plot(decTree)

# Evaluate fitting using model summary
# View model's summary
summary(decTree)

# Evaluamos el error del modelo de prediccion usando la matriz de confusion

cm = as.matrix(table(test_set[, 7], y_pred))
library(gmodels)
gmodels::CrossTable(test_set$car,
                    y_pred,
                    prop.chisq = FALSE,
                    prop.c     = FALSE,
                    prop.r     = FALSE,
                    dnn = c('actual default', 'predicted default'))

# Evaluamos el error del modelo de prediccion usando precision-recall y F1 score
precision = diag(cm) / colSums(cm)
recall = diag(cm) / rowSums(cm)
f1_score = ifelse(precision + recall == 0, 0, 2 * precision * recall / (precision + recall))


library(mlr)
dataset.task = makeClassifTask(id = "car", data = dataset, target = "car")
learningCurve = generateLearningCurveData(
  learners = c("classif.C50"),
  task = dataset.task,
  percs = seq(0.1, 1, by = 0.1),
  resampling = makeResampleDesc(method = "CV", iters = 5, predict = 'both'),
  measures = list(setAggregation(acc, train.mean), setAggregation(acc, test.mean)),
  show.info = TRUE)
plotLearningCurve(learningCurve, facet = 'learner')

# Predecimos la decision del usuario usando el modelo
decision = list(persons=4, buying='med', maint='low',
                lug_boots='med', safety='med', doors=4)
# Debido a que las personas y las puertas son categoricas las tenemos que convertir a factor.
decision$doors = factor(decision$doors, levels = c(2, 3, 4, '5more'))
decision$persons = factor(decision$persons, levels = c(2, 4, 'more'))
pred_dec = predict(decTree, newdata = decision, type = 'class')
pred_dec