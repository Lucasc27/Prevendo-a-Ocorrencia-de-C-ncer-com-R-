# -------------------------------------------------------------------------
#
# [R] - Prevendo a Ocorrencia de Câncer - 15/06/2019
#
# Lucas Cesar Fernandes Ferreira
# WebSite : www.lucascesarfernandes.com.br
# Linkedin : https://www.linkedin.com/in/lucas-cesar-fernandes/
# E-mail : lucascesar270392@gmail.com
# Contato : (31) 9 8219-8765
# -------------------------------------------------------------------------
# :: DETALHES ::
# Nesse mini projeto vamos prever a ocorrência de diabetes e para isso temos dados
# históricos de pacientes, o dataset tem 569 observações e 31 variáveis,
# nossa variável target está como Benigno ou Maligno.
# -------------------------------------------------------------------------

# Instalando os pacotes caso já não estejam instalando
if (! "randomForest" %in% installed.packages()) install.packages("randomForest")
if (! "class" %in% installed.packages()) install.packages("class")
if (! "gmodels" %in% installed.packages()) install.packages("gmodels")
if (! "caret" %in% installed.packages()) install.packages("caret")
if (! "e1071" %in% installed.packages()) install.packages("e1071")
if (! "ggplot2" %in% installed.packages()) install.packages("ggplot2")
if (! "plotly" %in% installed.packages()) install.packages("plotly")
if (! "DMwR" %in% installed.packages()) install.packages("DMwR")

# Carregando Pacotes
library(randomForest)
library(class)
library(gmodels)
library(caret)
library(e1071)
library(ggplot2)
library(plotly)
library(DMwR)

# Definindo o diretório de trabalho
getwd()
setwd("C:/Users/lucas.a.ferreira/Documents/Projetos-em-R/OcorrenciaDeCancer")

# Carregando os dados
dados <- read.csv('dataset.csv', header = TRUE, sep = ',')
View(dados)
head(dados)

# Explorando os dados - Visualizando as variáveis
str(dados)

# Explorando os dados - Medidas de tendência central
summary(dados)

# Explorando os dados - Valores missing
sapply(dados, function(x) sum(is.na(x)))

# Vamos melhorar o nome da variável target e transforma-lá em factor
dados$diagnosis <- factor(dados$diagnosis, levels = c("B", "M"),
                          labels = c("Benigno", "Maligno"))
dados$diagnosis[1:100]

# Vamos apagar a coluna ID, pois ela é somente uma identificação única
dados$id <- NULL

# Verificando a proporção da variável target
round(prop.table(table(dados$diagnosis))*100, digits = 2)

# Os dados estão apresentando grande variação na escala, por esse motivo temos que aplicar
#         a normalização pois alguns algoritmos esperam que os dados estejam normalizados
summary(dados[c('radius_mean','texture_mean','perimeter_mean',
                'area_mean','smoothness_mean')])

# Aplicando a função de normalização no dataset
normalization <- function(x) {
  
  return ((x - min(x)) / (max(x) - min(x)))
  
}

dados_normalized <- as.data.frame(lapply(dados[2:31], normalization))
dados_normalized$diagnosis <- dados$diagnosis
View(dados_normalized)

# Criando a partição padrão do dataset
partition <- 0.75
set.seed(2019)
split <- createDataPartition(y = dados_normalized$diagnosis, p = partition, list = FALSE)

# Verificando as variáveis mais importantes
set.seed(2019)
correlation <- randomForest(diagnosis ~ .,
                            data = dados_normalized, importance = T); correlation
varImpPlot(correlation)

# Criando o modelo com o algoritmo RandomForest
dados_treino_rf <- dados_normalized[split,]
dados_teste_rf <- dados_normalized[-split,]

set.seed(2019)
model_rf <- randomForest(diagnosis ~ .
                  - symmetry_mean
                  - dimension_mean
                  - symmetry_se
                  - smoothness_se,
                  data = dados_treino_rf, importance = T);

model_rf_pred <- predict(model_rf, dados_teste_rf[,1:30])

confusionMatrix(dados_teste_rf$diagnosis, model_rf_pred)
rf_result = round((mean(model_rf_pred == dados_teste_rf$diagnosis)*100), digits = 2)

# Criando o modelo com o algoritmo KNN
dados_treino_knn <- dados_normalized[split,]
dados_teste_knn <- dados_normalized[-split,]

dados_treino_labels_knn <- dados_treino_knn[, 31]
dados_teste_labels_knn <- dados_teste_knn[, 31]

set.seed(2019)
modelo_knn <- knn(train = dados_treino_knn[-31], 
                  test = dados_teste_knn[-31],
                  cl = dados_treino_labels_knn, 
                  k = 20)

confusionMatrix(dados_teste_labels_knn, modelo_knn)
knn_result = round((mean(modelo_knn == dados_teste_labels_knn)*100), digits = 2)

# Criando o modelo com o algoritmo de Support Vector Machine (SVM)
dados_treino_svm <- dados_normalized[split,]
dados_teste_svm <- dados_normalized[-split,]

set.seed(2019)
modelo_svm <- svm(diagnosis ~ ., 
                  data = dados_treino_svm, 
                  type = 'C-classification', 
                  kernel = 'radial')

modelo_svm_pred <- predict(modelo_svm, dados_teste_svm[,1:30])

confusionMatrix(dados_teste_svm$diagnosis, modelo_svm_pred)
svm_result = round((mean(modelo_svm_pred == dados_teste_svm$diagnosis)*100), digits = 2)

# Criando o modelo com o Generalized linear model (glm)
dados_treino_glm <- dados_normalized[split,]
dados_teste_glm <- dados_normalized[-split,]

modelo_glm <- glm(diagnosis ~ ., data = dados_treino_glm, family = 'binomial')

dados_teste_glm$diagnosis_character <- as.character(dados_teste_glm$diagnosis)
dados_teste_glm$diagnosis_character[dados_teste_glm$diagnosis_character=="Maligno"] <- "0"
dados_teste_glm$diagnosis_character[dados_teste_glm$diagnosis_character=="Benigno"] <- "1"

results_glm_pred <- predict(modelo_glm, dados_teste_glm[,1:30], type = 'response')
results_glm_pred <- ifelse(results_glm_pred > 0.5,1,0)
prob.error <- mean(results_glm_pred != dados_teste_glm$diagnosis_character)

CrossTable(dados_teste_glm$diagnosis, results_glm_pred, prop.chisq = FALSE)
glm_result = round((prob.error *100), digits = 2)

# Criando o modelo com o algoritmo KNN com os dados padronizados
#                   (z-score padronizado)usando a função scale() 
dados_z <- as.data.frame(scale(dados[-1]))
dados_z$diagnosis <- dados$diagnosis
summary(dados_z[c('radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean')])

set.seed(2019)
split <- createDataPartition(y = dados_z$diagnosis, p = partition, list = FALSE)

dados_treino_score_knn <- dados_z[split,]
dados_teste_score_knn <- dados_z[-split,]

dados_treino_labels_score_knn <- dados_treino_score_knn[, 31]
dados_teste_labels_score_knn <- dados_teste_score_knn[, 31]

set.seed(2019)
modelo_score_knn <- knn(train = dados_treino_score_knn[-31], 
                  test = dados_teste_score_knn[-31],
                  cl = dados_treino_labels_score_knn, 
                  k = 20)

confusionMatrix(dados_teste_labels_score_knn, modelo_score_knn)
knn_score_result = round((mean(modelo_score_knn == dados_teste_labels_score_knn)*100), digits = 2)

# Verificando as acurácias de cada modelo
models_results = data.frame(
  Algoritmo = c('RF','KNN','SVM','GLM','KNN_zScore'),
  Resultado = c(rf_result, knn_result, svm_result, glm_result, knn_score_result))

models_results <- data.frame(
  Algorithm = c('RF','RF','RF',
                'KNN','KNN','KNN',
                'SVM','SVM','SVM',
                'GLM','GLM','GLM',
                'KNN_zScore','KNN_zScore','KNN_zScore'),
  Partition = c('0.65%','0.75%','0.85%',
                '0.65%','0.75%','0.85%',
                '0.65%','0.75%','0.85%',
                '0.65%','0.75%','0.85%',
                '0.65%','0.75%','0.85%'),
  Results = c(96.46,97.89,97.62,
              97.18,97.18,97.18,
              97.98,97.89,97.62,
              95.96,95.77,95.24,
              96.48,96.48,96.48)
)
View(models_results)
# Plotando o resultado dos modelos
ggplot(data = models_results,
       aes(x=Algorithm, y=Results, group=Partition, colour=Partition)) +
  geom_line() +
  geom_point(shape=3)

# Criand novo modelo usando o método SMOTE para balancear a variável target
dados_smote <- read.csv('dataset.csv', header = TRUE, sep = ',')
dados_smote$id <- NULL
dados_smote$diagnosis <- factor(dados_smote$diagnosis, levels = c("B", "M"),
                                labels = c("Benigno", "Maligno"))

dados_normalized_smote <- as.data.frame(lapply(dados_smote[2:31], normalization))
dados_normalized_smote$diagnosis <- dados_smote$diagnosis

set.seed(2019)
dados_normalized_smote <- SMOTE(diagnosis ~ ., dados_normalized_smote,
                                perc.over = 190, k = 5, perc.under = 190)

round(prop.table(table(dados_normalized_smote$diagnosis))*100, digits = 2)
dim(dados_normalized_smote)

set.seed(2019)
split <- createDataPartition(y = dados_normalized_smote$diagnosis, p = 0.7, list = FALSE)

dados_treino_svm_smote <- dados_normalized_smote[split,]
dados_teste_svm_smote <- dados_normalized_smote[-split,]

set.seed(2019)
modelo_svm_smote <- svm(diagnosis ~ ., 
                    data = dados_treino_svm_smote, 
                    type = 'C-classification', 
                    kernel = 'radial')

modelo_svm_smote_pred <- predict(modelo_svm_smote, dados_teste_svm_smote[,1:30])

confusionMatrix(dados_teste_svm_smote$diagnosis, modelo_svm_smote_pred)

































