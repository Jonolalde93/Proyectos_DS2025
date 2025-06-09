# Cargar librerías necesarias
library(dplyr) 
library(ggplot2) 
library(plotly)
library(caret) 
library(pROC)  
library(randomForest)
library(rpart)
library(e1071)
library(coin)
library(shiny)
library(DT)
library(tidyr)


# Cargar los datos desde un archivo CSV
file_path <- file.choose()  # Seleccionar el archivo CSV
data <- read.csv(file_path)

# Manejar valores faltantes en Medication_History (si es necesario)
data$Medication_History <- ifelse(is.na(data$Medication_History), "No", data$Medication_History)

# Eliminar la variable Dosage.in.mg ya que no es necesaria para el análisis
data <- subset(data, select = -c(Dosage.in.mg))

# Eliminar la variable Prescription ya que no es necesaria para el análisis
data <- subset(data, select = -c(Prescription))

# Eliminar la variable Cognitive test scores ya que no es necesaria para el análisis
data <- subset(data, select = -c(Cognitive_Test_Scores))

# Convertir variables categóricas a factores en todo el dataset 'data'
data$Education_Level <- factor(data$Education_Level)
data$Dominant_Hand <- factor(data$Dominant_Hand)
data$Gender <- factor(data$Gender)
data$Family_History <- factor(data$Family_History)
data$Smoking_Status <- factor(data$Smoking_Status)
data$APOE_ε4 <- factor(data$APOE_ε4)
data$Physical_Activity <- factor(data$Physical_Activity)
data$Depression_Status <- factor(data$Depression_Status)
data$Medication_History <- factor(data$Medication_History)
data$Nutrition_Diet <- factor(data$Nutrition_Diet)
data$Sleep_Quality <- factor(data$Sleep_Quality)
data$Chronic_Health_Conditions <- factor(data$Chronic_Health_Conditions)
data$Diabetic <- factor(data$Diabetic)

# División de los datos
set.seed(40)
train_index <- sample(1:nrow(data), 0.7 * nrow(data))
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

# Convertir la variable de respuesta a factor
train_data$Dementia <- factor(train_data$Dementia)
test_data$Dementia <- factor(test_data$Dementia)

# Modelo Random Forest

# Ajustar el modelo random forest
set.seed(40)
dementia_rf <- randomForest(Dementia ~ ., data = train_data, ntree = 500)

# Realizar predicciones para los datos de prueba
pred_rf <- predict(dementia_rf, newdata = test_data)

# Obtener probabilidades predichas para los datos de prueba
prob_pred_rf <- predict(dementia_rf, newdata = test_data, type = "prob")

# Comprobación de los nombres de las columnas en prob_pred_rf 
colnames(prob_pred_rf) <- make.names(colnames(prob_pred_rf))

# Calcular la matriz de confusión
conf_mat_rf <- confusionMatrix(pred_rf, test_data$Dementia)

# Calcular métricas de rendimiento
accuracyrf <- conf_mat_rf$overall['Accuracy']
precisionrf <- conf_mat_rf$byClass['Pos Pred Value']
recallrf <- conf_mat_rf$byClass['Sensitivity']

# Calcular el ROC y AUC
roc_rf <- roc(test_data$Dementia, prob_pred_rf[, "X1"])
auc_valuerf <- auc(roc_rf)

# Calcular el intervalo de confianza para AUC
ci_aucrf <- ci.auc(roc_rf)

# Mostrar los resultados de las métricas para "Random Forest"
cat("Exactitud:", accuracyrf, "\n")
cat("Precisión:", precisionrf, "\n")
cat("Sensibilidad:", recallrf, "\n")
cat("Área bajo la curva (AUC):", auc_valuerf, "\n")
cat("Intervalo de Confianza (AUC):", ci_aucrf, "\n")

# Modelo Árbol de Decisión
set.seed(40)
dementia_tree <- rpart(Dementia ~ ., data = train_data, method = "class")
pred_tree <- predict(dementia_tree, newdata = test_data, type = "class")
prob_pred_tree <- predict(dementia_tree, newdata = test_data, type = "prob")
conf_mat_tree <- confusionMatrix(pred_tree, test_data$Dementia)
roc_tree <- roc(test_data$Dementia, prob_pred_tree[, "1"])

# Modelo SVM
set.seed(40)
model_svm <- svm(Dementia ~ ., data = train_data, probability = TRUE)
pred_svm <- predict(model_svm, newdata = test_data, probability = TRUE)
prob_pred_svm <- attr(pred_svm, "probabilities")[,2]
conf_mat_svm <- confusionMatrix(factor(pred_svm, levels = levels(test_data$Dementia)), test_data$Dementia)
roc_svm <- roc(test_data$Dementia, prob_pred_svm)

# Calcular métricas para cada modelo
metrics_rf <- list(
  accuracy = conf_mat_rf$overall['Accuracy'],
  precision = conf_mat_rf$byClass['Pos Pred Value'],
  recall = conf_mat_rf$byClass['Sensitivity'],
  auc = auc(roc_rf),
  ci_auc = ci.auc(roc_rf)
)

metrics_tree <- list(
  accuracy = conf_mat_tree$overall['Accuracy'],
  precision = conf_mat_tree$byClass['Pos Pred Value'],
  recall = conf_mat_tree$byClass['Sensitivity'],
  auc = auc(roc_tree),
  ci_auc = ci.auc(roc_tree)
)

metrics_svm <- list(
  accuracy = conf_mat_svm$overall['Accuracy'],
  precision = conf_mat_svm$byClass['Pos Pred Value'],
  recall = conf_mat_svm$byClass['Sensitivity'],
  auc = auc(roc_svm),
  ci_auc = ci.auc(roc_svm)
)

# Preparar las estadísticas descriptivas
desc_stats_df <- data %>%
  summarise(across(where(is.numeric), list(
    Mean = mean,
    Median = median,
    SD = sd,
    Q1 = ~quantile(., 0.25),
    Q3 = ~quantile(., 0.75)
  ), .names = "{col}_{fn}")) %>%
  pivot_longer(cols = everything(), names_to = "Statistic", values_to = "Value")

# Calcula moda para todas las variables categóricas
freq_mode_df <- data %>%
  select(where(is.factor)) %>%
  summarise(across(everything(), list(
    Mode = ~names(sort(table(.), decreasing = TRUE)[1])
  ), .names = "{col}_{fn}")) %>%
  pivot_longer(cols = everything(), names_to = "Variable", values_to = "Value")

# Realizar pruebas estadísticas
realizar_pruebas_estadisticas <- function(data, categorical_vars, numeric_vars) {
  # Chi-cuadrado
  chi_square_results <- list()
  residuals_results <- list()
  
  for (var in categorical_vars) {
    if (var != "Dementia") {
      chi_test <- chisq.test(table(data[[var]], data$Dementia))
      chi_square_results[[var]] <- chi_test
      residuals <- chi_test$stdres
      residuals_results[[var]] <- residuals
    }
  }
  
  # Pruebas t de Student y Mann-Whitney
  t_test_results <- list()
  mann_whitney_results <- list()
  
  for (var in numeric_vars) {
    if (length(data[[var]]) == length(data$Dementia)) {
      shapiro_test <- shapiro.test(data[[var]])
      
      if (shapiro_test$p.value > 0.05) {
        t_test <- t.test(data[[var]] ~ data$Dementia)
        t_test_results[[var]] <- t_test
      } else {
        mann_whitney_test <- wilcox.test(data[[var]] ~ data$Dementia)
        mann_whitney_results[[var]] <- mann_whitney_test
      }
    }
  }
  
  return(list(
    chi_square = chi_square_results,
    residuals = residuals_results,
    t_test = t_test_results,
    mann_whitney = mann_whitney_results
  ))
}

# Definir la UI
ui <- fluidPage(
  titlePanel("Dashboard TFM"),
  
  sidebarLayout(
    sidebarPanel(
      # Controles para las variables de entrada
      numericInput("AlcoholLevel", "Alcohol en sangre en g/l :", value = 0.15),
      numericInput("Age", "Edad:", value = 62),
      numericInput("HeartRate", "Frecuencia Cardiaca:", value = 60),
      numericInput("BloodOxygenLevel", "Nivel de O2 en sangre:", value = 99),
      numericInput("BodyTemperature", "Temperatura corporal:", value = 36),
      numericInput("Weight", "Peso:", value = 65),
      numericInput("MRI_Delay", "Retraso del MRI (en dias):", value = 65),
      selectInput("Gender", "Género:", choices = c("Male", "Female"), selected = "Male"),
      selectInput("Smoking_Status", "Fumador:", choices = c("Current Smoker", "Never Smoked", "Former Smoker"), selected = "Never Smoked"),
      selectInput("Diabetic", "Diabetes:", choices = c(1, 0), selected = 1),
      selectInput("Dominant_Hand", "Mano Dominante:", choices = c("Left", "Right"), selected = "Left"),
      selectInput("Family_History", "Historial Familiar:", choices = c("Yes", "No"), selected = "Yes"),
      selectInput("APOE_ε4", "Presencia de alelo APOE-ε4:", choices = c("Positive", "Negative"), selected = "Positive"),
      selectInput("Physical_Activity", "Actividad física:", choices = c("Moderate Activity", "Mild Activity", "Sedentary"), selected = "Sedentary"),
      selectInput("Depression_Status", "Depresión:", choices = c("Yes", "No"), selected = "Yes"),
      selectInput("Medication_History", "Medicación previa:", choices = c("Yes", "No"), selected = "Yes"),
      selectInput("Nutrition_Diet", "Dieta:", choices = c("Mediterranean Diet", "Balanced Diet", "Low-carb Diet"), selected = "Balanced Diet"),
      selectInput("Sleep_Quality", "Calidad del sueño:", choices = c("Good", "Poor"), selected = "Poor"),
      selectInput("Chronic_Health_Conditions", "Condiciones de salud crónicas:", choices = c("Heart Disease", "Hypertension", "Diabetes", "None"), selected = "Diabetes"),
      selectInput("Education_Level", "Nivel Educativo:", choices = c("Primary School", "Secondary School", "Diploma/Degree"), selected = "Primary School"),
      actionButton("predict_button", "Hacer predicción")
    ),
    
    mainPanel(
      mainPanel(
        tabsetPanel(
          tabPanel("Estadísticas Descriptivas",
                   h4("Estadísticas Descriptivas de Variables Numéricas"),
                   DTOutput("numeric_stats_output"),
                   h4("Moda de las Variables Categóricas"),
                   DTOutput("categorical_stats_output")
        ),
        tabPanel("Gráficos descriptivos", 
                 plotlyOutput("alcohol_plot"),
                 plotlyOutput("age_plot"),
                 plotlyOutput("heart_plot"),
                 plotlyOutput("oxygen_plot"),
                 plotlyOutput("temperature_plot"),
                 plotlyOutput("weight_plot"),
                 plotlyOutput("mri_plot"),
                 plotlyOutput("gender_plot"),
                 plotlyOutput("gender_frecuencias"),
                 plotlyOutput("smoking_plot"),
                 plotlyOutput("smoking_frecuencias"),
                 plotlyOutput("diabetic_plot"),
                 plotlyOutput("diabetic_frecuencias"),
                 plotlyOutput("hand_plot"),
                 plotlyOutput("hand_frecuencias"),
                 plotlyOutput("family_plot"),
                 plotlyOutput("family_frecuencias"),
                 plotlyOutput("apoe_ε4_plot"),
                 plotlyOutput("apoe_ε4_frecuencias"),
                 plotlyOutput("activity_plot"),
                 plotlyOutput("activity_frecuencias"),
                 plotlyOutput("depression_plot"),
                 plotlyOutput("depression_frecuencias"),
                 plotlyOutput("medication_plot"),
                 plotlyOutput("medication_frecuencias"),
                 plotlyOutput("diet_plot"),
                 plotlyOutput("diet_frecuencias"),
                 plotlyOutput("sleep_plot"),
                 plotlyOutput("sleep_frecuencias"),
                 plotlyOutput("condition_plot"),
                 plotlyOutput("condition_frecuencias"),
                 plotlyOutput("education_plot"),
                 plotlyOutput("education_frecuencias")
        ),
        tabPanel("Modelos Predictivos",
                 h4("Métricas Random Forest"),
                 verbatimTextOutput("metrics_rf_output"),
                 h4("Métricas Árbol de Decisión"),
                 verbatimTextOutput("metrics_tree_output"),
                 h4("Métricas SVM"),
                 verbatimTextOutput("metrics_svm_output"),
                 plotlyOutput("AUC1_plot"),
                 plotlyOutput("AUC2_plot"),
                 plotlyOutput("AUC3_plot")
        ),
        tabPanel("Predicción",
                 h4("Resultados de la predicción"),
                 verbatimTextOutput("prediction_output")
        ),
        tabPanel("Pruebas estadísticas",
                 h4("Resultados Chi-Cuadrado"),
                 verbatimTextOutput("chi_square_output"),
                 h4("Pruebas t de Student"),
                 verbatimTextOutput("t_test_output"),
                 h4("Pruebas Mann-Whitney"),
                 verbatimTextOutput("mann_whitney_output")
        ),
      )
    )
  )
)
)
server <- function(input, output) {
  # Realizar predicción basada en la entrada del usuario
  observeEvent(input$predict_button, {
    tryCatch({
      # Verificar que todas las entradas sean válidas y no vacías
      validate(
        need(!is.na(as.numeric(input$AlcoholLevel)) && input$AlcoholLevel != "", "AlcoholLevel es inválido"),
        need(!is.na(as.numeric(input$Age)) && input$Age != "", "Age es inválido"),
        need(!is.na(as.numeric(input$HeartRate)) && input$HeartRate != "", "Heart_Rate es inválido"),
        need(!is.na(as.numeric(input$BloodOxygenLevel)) && input$BloodOxygenLevel != "", "BloodOxygenLevel es inválido"),
        need(!is.na(as.numeric(input$BodyTemperature)) && input$BodyTemperature != "", "BodyTemperature es inválido"),
        need(!is.na(as.numeric(input$Weight)) && input$Weight != "", "Weight es inválido"),
        need(!is.na(as.numeric(input$MRI_Delay)) && input$MRI_Delay != "", "MRI_Delay es inválido"),
        need(input$Gender %in% levels(data$Gender), "Gender es inválido"),
        need(input$Smoking_Status %in% levels(data$Smoking_Status), "Smoking_Status es inválido"),
        need(input$Diabetic %in% levels(data$Diabetic), "Diabetic es inválido"),
        need(input$Dominant_Hand %in% levels(data$Dominant_Hand), "Dominant_Hand es inválido"),
        need(input$Family_History %in% levels(data$Family_History), "Family_History es inválido"),
        need(input$APOE_ε4 %in% levels(data$APOE_ε4), "APOE_ε4 es inválido"),
        need(input$Physical_Activity %in% levels(data$Physical_Activity), "Physical_Activity es inválido"),
        need(input$Depression_Status %in% levels(data$Depression_Status), "Depression_Status es inválido"),
        need(input$Medication_History %in% levels(data$Medication_History), "Medication_History es inválido"),
        need(input$Nutrition_Diet %in% levels(data$Nutrition_Diet), "Nutrition_Diet es inválido"),
        need(input$Sleep_Quality %in% levels(data$Sleep_Quality), "Sleep_Quality es inválido"),
        need(input$Chronic_Health_Conditions %in% levels(data$Chronic_Health_Conditions), "Chronic_Health_Conditions es inválido"),
        need(input$Education_Level %in% levels(data$Education_Level), "Education_Level es inválido")
      )
      
      # Crear el dataframe asegurando que todos los niveles de factores coinciden
      user_data <- data.frame(
        AlcoholLevel = as.numeric(input$AlcoholLevel),
        Age = as.numeric(input$Age),
        HeartRate = as.numeric(input$HeartRate),
        BloodOxygenLevel = as.numeric(input$BloodOxygenLevel),
        BodyTemperature = as.numeric(input$BodyTemperature),
        Weight = as.numeric(input$Weight),
        MRI_Delay = as.numeric(input$MRI_Delay),
        Gender = factor(input$Gender, levels = levels(data$Gender)),
        Smoking_Status = factor(input$Smoking_Status, levels = levels(data$Smoking_Status)),
        Diabetic = factor(input$Diabetic, levels = levels(data$Diabetic)),
        Dominant_Hand = factor(input$Dominant_Hand, levels = levels(data$Dominant_Hand)),
        Family_History = factor(input$Family_History, levels = levels(data$Family_History)),
        APOE_ε4 = factor(input$APOE_ε4, levels = levels(data$APOE_ε4)),
        Physical_Activity = factor(input$Physical_Activity, levels = levels(data$Physical_Activity)),
        Depression_Status = factor(input$Depression_Status, levels = levels(data$Depression_Status)),
        Medication_History = factor(input$Medication_History, levels = levels(data$Medication_History)),
        Nutrition_Diet = factor(input$Nutrition_Diet, levels = levels(data$Nutrition_Diet)),
        Sleep_Quality = factor(input$Sleep_Quality, levels = levels(data$Sleep_Quality)),
        Chronic_Health_Conditions = factor(input$Chronic_Health_Conditions, levels = levels(data$Chronic_Health_Conditions)),
        Education_Level = factor(input$Education_Level, levels = levels(data$Education_Level))
      )
      
      # Realizar predicciones
      rf_prediction <- predict(dementia_rf, newdata = user_data)
      tree_prediction <- predict(dementia_tree, newdata = user_data, type = "class")
      svm_prediction <- predict(model_svm, newdata = user_data)
      # Convertir las predicciones a texto descriptivo
      
      rf_text <- ifelse(rf_prediction == 1, "Segun el modelo Random Forest, hay riesgo de desarrollar demencia", "Segun el modelo Random Forest, no hay riesgo de desarrollar demencia")
      tree_text <- ifelse(tree_prediction == 1, "Segun el modelo Decision Tree, hay riesgo de desarrollar demencia", "Segun el modelo Decision Tree, no hay riesgo de desarrollar demencia")
      svm_text <- ifelse(svm_prediction == 1, "Segun el modelo SVM, hay riesgo de desarrollar demencia", "Segun el modelo SVM, no hay riesgo de desarrollar demencia")
      
      # Mostrar resultados
      output$prediction_output <- renderPrint({
        list(
          Random_Forest = rf_text,
          Decision_Tree = tree_text,
          SVM = svm_text
        )
      })
    }, error = function(e) {
      output$prediction_output <- renderPrint({
        paste("Error en la predicción:", e$message)
      })
    })
  })
  # Estadísticas descriptivas para variables numéricas y categoricas
  output$numeric_stats_output <- renderDT({
    desc_stats_df
  }, options = list(pageLength = 10, autoWidth = TRUE))
  
  # Calcula y muestra las modas para las variables categóricas
  output$categorical_stats_output <- DT::renderDT({
    freq_mode_df
  }, options = list(pageLength = 10, autoWidth = TRUE))

  # Gráficos descriptivos
  output$alcohol_plot <- renderPlotly({
    alcohol_ranges <- cut(data$AlcoholLevel, 
                          breaks = c(0.00, 0.05, 0.1, 0.15, 0.2), 
                          labels = c("0-0.05", "0.05-0.1", "0.1-0.15", "0.15-0.2"))
    data$AlcoholLevel <- factor(alcohol_ranges)
    plot_alcohol <- ggplot(data, aes(x = Dementia, fill = AlcoholLevel)) +
      geom_bar(position = "dodge", stat = "count") +
      labs(title = "Alcohol") +
      scale_x_discrete(labels = c("0" = "No demencia", "1" = "Demencia")) +
      scale_fill_discrete(name = "Alcohol Level", labels = c("0-0.05", "0.05-0.1", "0.1-0.15", "0.15-0.2")) +
      theme_minimal()
    plot_alcohol_interactive <- ggplotly(plot_alcohol)
    plot_alcohol_interactive <- plot_alcohol_interactive %>% 
      layout(
        xaxis = list(
          tickvals = c("0", "1"),  
          ticktext = c("No demencia", "Demencia")  
        )
      )
    plot_alcohol_interactive
  })
  output$age_plot <- renderPlotly({
    plot_age <- ggplot(data, aes(x = Dementia, fill = factor(Age))) +
      geom_bar(position = "dodge", stat = "count") +
      labs(title = "Edad") +
      scale_x_discrete(labels = c("0" = "No Demencia", "1" = "Demencia")) +
      scale_fill_discrete(name = "Age") +
      theme_minimal()
    plot_age_interactive <- ggplotly(plot_age)
    plot_age_interactive <- plot_age_interactive %>% 
      layout(
        xaxis = list(
          tickvals = c("0", "1"),  
          ticktext = c("No demencia", "Demencia")  
        )
      )
    plot_age_interactive
  })
  output$heart_plot <- renderPlotly({
    plot_heart <- ggplot(data, aes(x = Dementia, fill = factor(HeartRate))) +
      geom_bar(position = "dodge", stat = "count") +
      labs(title = "Heart Rate") +
      scale_x_discrete(labels = c("0" = "No Demencia", "1" = "Demencia")) + 
      scale_fill_discrete(name = "Heart Rate") +
      theme_minimal()
    plot_heartrate_interactive <- ggplotly(plot_heart)
    plot_heartrate_interactive <- plot_heartrate_interactive %>% 
      layout(
        xaxis = list(
          tickvals = c("0", "1"),  
          ticktext = c("No demencia", "Demencia")  
        )
      )
    plot_heartrate_interactive
  })
  output$oxygen_plot <- renderPlotly({
    oxygen_ranges <- cut(data$BloodOxygenLevel, 
                         breaks = c(90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100), 
                         labels = c("90-91", "91-92", "92-93", "93-94", "94-95", "95-96", "96-97", "97-98", "98-99", "99-100"))
    data$BloodOxygenLevel <- factor(oxygen_ranges, levels = c("90-91", "91-92", "92-93", "93-94", "94-95", "95-96", "96-97", "97-98", "98-99", "99-100"))
    plot_oxygen <- ggplot(data, aes(x = Dementia, fill = BloodOxygenLevel)) +
      geom_bar(position = "dodge", stat = "count") +
      labs(title = "Blood Oxygen") +
      scale_x_discrete(labels = c("0" = "No Demencia", "1" = "Demencia")) +
      scale_fill_discrete(name = "Blood Oxygen Levels") +
      theme_minimal()
    plot_oxygen_interactive <- ggplotly(plot_oxygen)
    plot_oxygen_interactive <- plot_oxygen_interactive %>%
      layout(
        xaxis = list(
          tickvals = c("0", "1"),  
          ticktext = c("No demencia", "Demencia")  
        )
      )
    plot_oxygen_interactive
  })
  output$temperature_plot <- renderPlotly({
    temperature_ranges <- cut(data$BodyTemperature, 
                              breaks = c(36, 36.1, 36.2, 36.3, 36.4, 36.5, 36.6, 36.7, 36.8, 36.9, 37, 37.1, 37.2, 37.3, 37.4, 37.5),
                              labels = c("36-36.1", "36.1-36.2", "36.2-36.3", "36.3-36.4", "36.4-36.5", "36.5-36.6", "36.6-36.7", "36.7-36.8", "36.8-36.9", "36.9-37", "37-37.1", "37.1-37.2", "37.2-37.3", "37.3-37.4", "37.4-37.5"))
    data$BodyTemperature <- factor(temperature_ranges)
    plot_temperature <- ggplot(data, aes(x = Dementia, fill = BodyTemperature)) +
      geom_bar(position = "dodge", stat = "count") +
      labs(title = "Body Temperature") +
      scale_x_discrete(labels = c("0" = "No demencia", "1" = "Demencia")) +
      scale_fill_discrete(name = "Body Temperature", labels = c("36-36.1", "36.1-36.2", "36.2-36.3", "36.3-36.4", "36.4-36.5", "36.5-36.6", "36.6-36.7", "36.7-36.8", "36.8-36.9", "36.9-37", "37-37.1", "37.1-37.2", "37.2-37.3", "37.3-37.4", "37.4-37.5")) +
      theme_minimal()
    plot_temperature_interactive <- ggplotly(plot_temperature)
    plot_temperature_interactive <- plot_temperature_interactive %>% 
      layout(
        xaxis = list(
          tickvals = c("0", "1"),  
          ticktext = c("No demencia", "Demencia")  
        )
      )
    plot_temperature_interactive
  })
  output$weight_plot <- renderPlotly({
    weight_ranges <- cut(data$Weight, 
                         breaks = c(50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100),
                         labels = c("50-55", "55-60", "60-65", "65-70", "70-75", "75-80", "80-85", "85-90", "90-95", "95-100"))
    data$Weight <- factor(weight_ranges)
    plot_weight <- ggplot(data, aes(x = Dementia, fill = Weight)) +
      geom_bar(position = "dodge", stat = "count") +
      labs(title = "Weight") +
      scale_x_discrete(labels = c("0" = "No demencia", "1" = "Demencia")) +
      scale_fill_discrete(name = "Weight", labels = c("50-55", "55-60", "60-65", "65-70", "70-75", "75-80", "80-85", "85-90", "90-95", "95-100")) +
      theme_minimal()
    plot_weight_interactive <- ggplotly(plot_weight)
    plot_weight_interactive <- plot_weight_interactive %>% 
      layout(
        xaxis = list(
          tickvals = c("0", "1"),  
          ticktext = c("No demencia", "Demencia")  
        )
      )
    plot_weight_interactive
  })
  output$mri_plot <- renderPlotly({
    mri_ranges <- cut(data$MRI_Delay, 
                      breaks = c(0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60),
                      labels = c("0-5", "5-10", "10-15", "15-20", "20-25", "25-30", "30-35", "35-40", "40-45", "45-50", "50-55", "55-60"))
    data$MRI_Delay <- factor(mri_ranges)
    plot_mri <- ggplot(data, aes(x = Dementia, fill = MRI_Delay)) +
      geom_bar(position = "dodge", stat = "count") +
      labs(title = "MRI Delay") +
      scale_x_discrete(labels = c("0" = "No demencia", "1" = "Demencia")) +
      scale_fill_discrete(name = "MRI Delay", labels = c("0-5", "5-10", "10-15", "15-20", "20-25", "25-30", "30-35", "35-40", "40-45", "45-50", "50-55", "55-60")) +
      theme_minimal()
    plot_mri_interactive <- ggplotly(plot_mri)
    plot_mri_interactive <- plot_mri_interactive %>% 
      layout(
        xaxis = list(
          tickvals = c("0", "1"),  
          ticktext = c("No demencia", "Demencia")  
        )
      )
    plot_mri_interactive
  })
  output$gender_plot <- renderPlotly({
    gender_plot <- ggplot(data, aes(x = Dementia, fill = Gender)) +
      geom_bar(position = "dodge", stat = "count") +
      labs(title = "Gender") + 
      scale_x_discrete(labels = c("0" = "No demencia", "1" = "Demencia")) +
      scale_fill_discrete(name = "Gender", labels = c("Male", "Female")) +
      theme_minimal()
    plot_gender_interactive <- ggplotly(gender_plot)
    plot_gender_interactive <- plot_gender_interactive %>% 
      layout(
        xaxis = list(
          tickvals = c("0", "1"),  
          ticktext = c("No demencia", "Demencia")  
        )
      )
    plot_gender_interactive
  })
  output$gender_frecuencias <- renderPlotly({
    plot_gender2 <- ggplot(data, aes(x = Gender, fill = (Dementia))) +
      geom_bar(position = "dodge", stat = "count") +
      labs(title = "Gender Frecuencias") +
      scale_x_discrete(labels = c("0" = "No demencia", "1" = "Demencia")) +
      theme_minimal()
    plot_gender_frecuencias <- ggplotly(plot_gender2)
    plot_gender_frecuencias <- plot_gender_frecuencias %>% 
      layout(
        xaxis = list(
          tickvals = c("0", "1"),  
          ticktext = c("No demencia", "Demencia")  
        )
      )
    plot_gender_frecuencias
  })
  output$smoking_plot <- renderPlotly({
    plot_smoking <- ggplot(data, aes(x = Dementia, fill = Smoking_Status)) +
      geom_bar(position = "dodge", stat = "count") +
      labs(title = "Smoking Status") +
      scale_x_discrete(labels = c("0" = "No demencia", "1" = "Demencia")) +
      scale_fill_discrete(name = "Smoking Status", labels = c("Current Smoker", "Former Smoker", "Never Smoked"))
    theme_minimal()
    plot_smoking_interactive <- ggplotly(plot_smoking)
    plot_smoking_interactive <- plot_smoking_interactive %>% 
      layout(
        xaxis = list(
          tickvals = c("0", "1"),  
          ticktext = c("No demencia", "Demencia")  
        )
      )
    plot_smoking_interactive
  })
    output$smoking_frecuencias <- renderPlotly({
      plot_smoking2 <- ggplot(data, aes(x = Smoking_Status, fill = (Dementia))) +
        geom_bar(position = "dodge", stat = "count") +
        labs(title = "Smoking_Status Frecuencias") +
        scale_x_discrete(labels = c("0" = "No demencia", "1" = "Demencia")) +
        theme_minimal()
      plot_smoking_frecuencias <- ggplotly(plot_smoking2)
      plot_smoking_frecuencias <- plot_smoking_frecuencias %>% 
        layout(
          xaxis = list(
            tickvals = c("0", "1"),  
            ticktext = c("No demencia", "Demencia")  
          )
        )
      plot_smoking_frecuencias
  })
  output$diabetic_plot <- renderPlotly({
    plot_diabetic <- ggplot(data, aes(x = Dementia, fill = factor(Diabetic))) +
      geom_bar(position = "dodge", stat = "count") +
      labs(title = "Diabetic") +
      scale_x_discrete(labels = c("0" = "No demencia", "1" = "Demencia")) +
      scale_fill_discrete(name = "Diabetic", labels = c("No Diabetic", "Diabetic")) +
      theme_minimal()
    plot_diabetic_interactive <- ggplotly(plot_diabetic)
    plot_diabetic_interactive <- plot_diabetic_interactive %>% 
      layout(
        xaxis = list(
          tickvals = c("0", "1"),  
          ticktext = c("No demencia", "Demencia")  
        )
      )
    plot_diabetic_interactive
  })
  output$diabetic_frecuencias <- renderPlotly({
    plot_diabetic2 <- ggplot(data, aes(x = factor(Diabetic), fill = (Dementia))) +
      geom_bar(position = "dodge", stat = "count") +
      labs(title = "Diabetic Frecuencias") +
      scale_x_discrete(labels = c("0" = "No demencia", "1" = "Demencia")) +
      theme_minimal()
    plot_diabetic_frecuencias <- ggplotly(plot_diabetic2)
    plot_diabetic_frecuencias <- plot_diabetic_frecuencias %>% 
      layout(
        xaxis = list(
          tickvals = c("0", "1"),  
          ticktext = c("No demencia", "Demencia")  
        )
      )
    plot_diabetic_frecuencias
  })
  output$hand_plot <- renderPlotly({
    plot_hand <- ggplot(data, aes(x = Dementia, fill = Dominant_Hand)) +
      geom_bar(position = "dodge", stat = "count") +
      labs(title = "Dominant Hand") +
      scale_x_discrete(labels = c("0" = "No demencia", "1" = "Demencia")) +
      scale_fill_discrete(name = "Dominant Hand", labels = c("Left", "Right")) +
      theme_minimal()
    plot_hand_interactive <- ggplotly(plot_hand)
    plot_hand_interactive <- plot_hand_interactive %>% 
      layout(
        xaxis = list(
          tickvals = c("0", "1"),  
          ticktext = c("No demencia", "Demencia")  
        )
      )
    plot_hand_interactive
  })
  output$hand_frecuencias <- renderPlotly({
    plot_hand2 <- ggplot(data, aes(x = Dominant_Hand, fill = (Dementia))) +
      geom_bar(position = "dodge", stat = "count") +
      labs(title = "Dominant Hand Frecuencias") +
      scale_x_discrete(labels = c("0" = "No demencia", "1" = "Demencia")) +
      theme_minimal()
    plot_hand_frecuencias <- ggplotly(plot_hand2)
    plot_hand_frecuencias <- plot_hand_frecuencias %>% 
      layout(
        xaxis = list(
          tickvals = c("0", "1"),  
          ticktext = c("No demencia", "Demencia")  
        )
      )
    plot_hand_frecuencias
  })
  output$family_plot <- renderPlotly({
  plot_family <- ggplot(data, aes(x = Dementia, fill = Family_History)) +
    geom_bar(position = "dodge", stat = "count") +
    labs(title = "Family History") +
    scale_x_discrete(labels = c("0" = "No demencia","1" = "Demencia")) +
    scale_fill_discrete(name = "Family History", labels = c("No", "Yes")) +
    theme_minimal()
  plot_family_interactive <- ggplotly(plot_family)
  plot_family_interactive <- plot_family_interactive %>% 
    layout(
      xaxis = list(
        tickvals = c("0", "1"),  
        ticktext = c("No demencia", "Demencia")  
      )
    )
  plot_family_interactive
  })
  output$family_frecuencias <- renderPlotly({
    plot_family2 <- ggplot(data, aes(x = Family_History, fill = (Dementia))) +
      geom_bar(position = "dodge", stat = "count") +
      labs(title = "Family History Frecuencias") +
      scale_x_discrete(labels = c("0" = "No demencia", "1" = "Demencia")) +
      theme_minimal()
    plot_family_frecuencias <- ggplotly(plot_family2)
    plot_family_frecuencias <- plot_family_frecuencias %>% 
      layout(
        xaxis = list(
          tickvals = c("0", "1"),  
          ticktext = c("No demencia", "Demencia")  
        )
      )
    plot_family_frecuencias
  })  
  output$apoe_ε4_plot <- renderPlotly({
    plot_apoe <- ggplot(data, aes(x = Dementia, fill = APOE_ε4)) +
      geom_bar(position = "dodge", stat = "count") +
      labs(title = "APOE_ε4") +
      scale_x_discrete(labels = c("0" = "No demencia", "1" = "Demencia")) +
      scale_fill_discrete(name = "ApoE4", labels = c("Negative", "Positive")) +
      theme_minimal()
    plot_apoe_interactive <- ggplotly(plot_apoe)
    plot_apoe_interactive <- plot_apoe_interactive %>% 
      layout(
        xaxis = list(
          tickvals = c("0", "1"),  
          ticktext = c("No demencia", "Demencia")  
        )
      )
    plot_apoe_interactive
  })
    output$apoe_ε4_frecuencias <- renderPlotly({
      plot_apoe_ε42 <- ggplot(data, aes(x = APOE_ε4, fill = (Dementia))) +
        geom_bar(position = "dodge", stat = "count") +
        labs(title = "APOE_ε4 Frecuencias") +
        scale_x_discrete(labels = c("0" = "No demencia", "1" = "Demencia")) +
        theme_minimal()
      plot_apoe_ε4_frecuencias <- ggplotly(plot_apoe_ε42)
      plot_apoe_ε4_frecuencias <- plot_apoe_ε4_frecuencias %>% 
        layout(
          xaxis = list(
            tickvals = c("0", "1"),  
            ticktext = c("No demencia", "Demencia")  
          )
        )
      plot_apoe_ε4_frecuencias
  })
  output$activity_plot <- renderPlotly({
    plot_activity <- ggplot(data, aes(x = Dementia, fill = Physical_Activity)) +
      geom_bar(position = "dodge", stat = "count") +
      labs(title = "Physical Activity") +
      scale_x_discrete(labels = c("No Demencia" = "0", "Demencia" = "1")) +
      scale_fill_discrete(name = "Physical Activity", labels = c("Mild Activity", "Moderate Activity", "Sedentary")) +
      theme_minimal()
    plot_activity_interactive <- ggplotly(plot_activity)
    plot_activity_interactive <- plot_activity_interactive %>% 
      layout(
        xaxis = list(
          tickvals = c("0", "1"),  
          ticktext = c("No demencia", "Demencia")  
        )
      )
    plot_activity_interactive
  })
  output$activity_frecuencias <- renderPlotly({
    plot_activity2 <- ggplot(data, aes(x = Physical_Activity, fill = (Dementia))) +
      geom_bar(position = "dodge", stat = "count") +
      labs(title = "Physical Activity Frecuencias") +
      scale_x_discrete(labels = c("0" = "No demencia", "1" = "Demencia")) +
      theme_minimal()
    plot_activity_frecuencias <- ggplotly(plot_activity2)
    plot_activity_frecuencias <- plot_activity_frecuencias %>% 
      layout(
        xaxis = list(
          tickvals = c("0", "1"),  
          ticktext = c("No demencia", "Demencia")  
        )
      )
    plot_activity_frecuencias
  })
  output$depression_plot <- renderPlotly({
    depression_plot <- ggplot(data, aes(x = Dementia, fill = Depression_Status)) +
      geom_bar(position = "dodge") +
      theme_minimal() +
      scale_x_discrete(labels = c("0" = "No demencia", "1" = "Demencia")) +
      scale_fill_discrete(name = "Depression Status", labels = c("No", "Yes")) +
      labs(title = "Distribución por Estado de Depresión")
    ggplotly(depression_plot)
  })
  output$depression_frecuencias <- renderPlotly({
    plot_depression2 <- ggplot(data, aes(x = Depression_Status, fill = (Dementia))) +
      geom_bar(position = "dodge", stat = "count") +
      labs(title = "Depression Status Frecuencias") +
      scale_x_discrete(labels = c("0" = "No demencia", "1" = "Demencia")) +
      theme_minimal()
    plot_depression_frecuencias <- ggplotly(plot_depression2)
    plot_depression_frecuencias <- plot_depression_frecuencias %>% 
      layout(
        xaxis = list(
          tickvals = c("0", "1"),  
          ticktext = c("No demencia", "Demencia")  
        )
      )
    plot_depression_frecuencias
  })
  output$medication_plot <- renderPlotly({
    plot_medication <- ggplot(data, aes(x = Dementia, fill = `Medication_History`)) +
      geom_bar(position = "dodge", stat = "count") +
      labs(title = "Medication History") +
      scale_x_discrete(labels = c("No Demencia" = "0", "Demencia" = "1")) +      
      scale_fill_discrete(name = "Medication History", labels = c("No", "Yes")) +
      theme_minimal()
    plot_medication_interactive <- ggplotly(plot_medication)
    plot_medication_interactive <- plot_medication_interactive %>% 
      layout(
        xaxis = list(
          tickvals = c("0", "1"),  
          ticktext = c("No demencia", "Demencia")  
        )
      )
    plot_medication_interactive
  })
  output$medication_frecuencias <- renderPlotly({
    plot_medication2 <- ggplot(data, aes(x = Medication_History, fill = (Dementia))) +
      geom_bar(position = "dodge", stat = "count") +
      labs(title = "Medication History Frecuencias") +
      scale_x_discrete(labels = c("0" = "No demencia", "1" = "Demencia")) +
      theme_minimal()
    plot_medication_frecuencias <- ggplotly(plot_medication2)
    plot_medication_frecuencias <- plot_medication_frecuencias %>% 
      layout(
        xaxis = list(
          tickvals = c("0", "1"),  
          ticktext = c("No demencia", "Demencia")  
        )
      )
    plot_medication_frecuencias
  })
  output$diet_plot <- renderPlotly({
    plot_diet <- ggplot(data, aes(x = Dementia, fill = Nutrition_Diet)) +
      geom_bar(position = "dodge", stat = "count") +
      labs(title = "Nutrition Diet") +
      scale_x_discrete(labels = c("No Demencia" = "0", "Demencia" = "1")) +
      scale_fill_discrete(name = "Nutrition Diet", labels = c("Balanced diet", "Low-Carb diet", "Mediterranean diet")) +
      theme_minimal()
    plot_diet_interactive <- ggplotly(plot_diet)
    plot_diet_interactive <- plot_diet_interactive %>% 
      layout(
        xaxis = list(
          tickvals = c("0", "1"),  
          ticktext = c("No demencia", "Demencia")  
        )
      )
    plot_diet_interactive  
  })
  output$diet_frecuencias <- renderPlotly({
    plot_diet2 <- ggplot(data, aes(x = Nutrition_Diet, fill = (Dementia))) +
      geom_bar(position = "dodge", stat = "count") +
      labs(title = "Nutrition Diet Frecuencias") +
      scale_x_discrete(labels = c("0" = "No demencia", "1" = "Demencia")) +
      theme_minimal()
    plot_diet_frecuencias <- ggplotly(plot_diet2)
    plot_diet_frecuencias <- plot_diet_frecuencias %>% 
      layout(
        xaxis = list(
          tickvals = c("0", "1"),  
          ticktext = c("No demencia", "Demencia")  
        )
      )
    plot_diet_frecuencias
  })
  output$sleep_plot <- renderPlotly({
    plot_sleep <- ggplot(data, aes(x = Dementia, fill = Sleep_Quality)) +
      geom_bar(position = "dodge", stat = "count") +
      labs(title = "Sleep Quality") +
      scale_x_discrete(labels = c("No Demencia" = "0", "Demencia" = "1")) +
      scale_fill_discrete(name = "Sleep Quality", labels = c("Good", "Poor")) +
      theme_minimal()
    plot_sleep_interactive <- ggplotly(plot_sleep)
    plot_sleep_interactive <- plot_sleep_interactive %>% 
      layout(
        xaxis = list(
          tickvals = c("0", "1"),  
          ticktext = c("No demencia", "Demencia")  
        )
      )
    plot_sleep_interactive
  })
  output$sleep_frecuencias <- renderPlotly({
    plot_sleep2 <- ggplot(data, aes(x = Sleep_Quality, fill = (Dementia))) +
      geom_bar(position = "dodge", stat = "count") +
      labs(title = "Sleep Quality Frecuencias") +
      scale_x_discrete(labels = c("0" = "No demencia", "1" = "Demencia")) +
      theme_minimal()
    plot_sleep_frecuencias <- ggplotly(plot_sleep2)
    plot_sleep_frecuencias <- plot_sleep_frecuencias %>% 
      layout(
        xaxis = list(
          tickvals = c("0", "1"),  
          ticktext = c("No demencia", "Demencia")  
        )
      )
    plot_sleep_frecuencias
  })
  output$condition_plot <- renderPlotly({
    plot_chronic <- ggplot(data, aes(x = Dementia, fill = `Chronic_Health_Conditions`)) +
      geom_bar(position = "dodge", stat = "count") +
      labs(title = "Chronic Health Condition") +
      scale_x_discrete(labels = c("No Demencia" = "0", "Demencia" = "1")) +
      scale_fill_discrete(name = "Chronic Health Conditions", labels = c("Diabetes", "Heart Disease", "Hypertension","None")) +
      theme_minimal()
    plot_chronic_interactive <- ggplotly(plot_chronic)
    plot_chronic_interactive <- plot_chronic_interactive %>% 
      layout(
        xaxis = list(
          tickvals = c("0", "1"),  
          ticktext = c("No demencia", "Demencia")  
        )
      )
    plot_chronic_interactive
  })
    output$condition_frecuencias <- renderPlotly({
      plot_condition2 <- ggplot(data, aes(x = Chronic_Health_Conditions, fill = (Dementia))) +
        geom_bar(position = "dodge", stat = "count") +
        labs(title = "Chronic Health Condition Frecuencias") +
        scale_x_discrete(labels = c("0" = "No demencia", "1" = "Demencia")) +
        theme_minimal()
      plot_condition_frecuencias <- ggplotly(plot_condition2)
      plot_condition_frecuencias <- plot_condition_frecuencias %>% 
        layout(
          xaxis = list(
            tickvals = c("0", "1"),  
            ticktext = c("No demencia", "Demencia")  
          )
        )
      plot_condition_frecuencias
  })
  output$education_plot <- renderPlotly({
    plot_education <- ggplot(data, aes(x = Dementia, fill = factor(Education_Level))) +
      geom_bar(position = "dodge", stat = "count") +
      labs(title = "Education Level Frecuencias") +
      scale_fill_discrete(name = "Education Level Frecuencias", labels = c("Diploma/Degree", "No School", "Primary School","Secondary School")) +
      theme_minimal()
    plot_education_interactive <- ggplotly(plot_education)
    plot_education_interactive
  })
  output$education_frecuencias <- renderPlotly({
    plot_education2 <- ggplot(data, aes(x = Education_Level, fill = (Dementia))) +
      geom_bar(position = "dodge", stat = "count") +
      labs(title = "Education Level") +
      scale_x_discrete(labels = c("0" = "No demencia", "1" = "Demencia")) +
      theme_minimal()
    plot_education_frecuencias <- ggplotly(plot_education2)
    plot_education_frecuencias <- plot_education_frecuencias %>% 
      layout(
        xaxis = list(
          tickvals = c("0", "1"),  
          ticktext = c("No demencia", "Demencia")  
        )
      )
    plot_education_frecuencias
  })
  output$AUC1_plot <- renderPlotly({
    roc_plot1 <- ggplot(data = data.frame(
      fpr = roc_rf$specificities,
      tpr = roc_rf$sensitivities
    ), aes(x = 1 - fpr, y = tpr)) +
      geom_line(color = "blue", size = 1) +
      geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
      annotate("text", x = 0.5, y = 0.2, label = paste("AUC =", round(metrics_rf$auc, 4)), size = 5) +
      labs(title = "Curva ROC para el modelo Random Forest",
           x = "1 - Especificidad",
           y = "Sensibilidad") +
      theme_minimal()
    roc_plot1_interactive <- ggplotly(roc_plot1)
    roc_plot1_interactive
})
output$AUC2_plot <- renderPlotly({
  roc_plot2 <- ggplot(data = data.frame(
    fpr = roc_tree$specificities,
    tpr = roc_tree$sensitivities
  ), aes(x = 1 - fpr, y = tpr)) +
    geom_line(color = "blue", size = 1) +
    geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
    annotate("text", x = 0.5, y = 0.2, label = paste("AUC =", round(metrics_tree$auc, 4)), size = 5) +
    labs(title = "Curva ROC para el modelo Decision tree",
         x = "1 - Especificidad",
         y = "Sensibilidad") +
    theme_minimal()
  roc_plot2_interactive <- ggplotly(roc_plot2)
  roc_plot2_interactive
})
output$AUC3_plot <- renderPlotly({
  roc_plot3 <- ggplot(data = data.frame(
    fpr = roc_svm$specificities,
    tpr = roc_svm$sensitivities
  ), aes(x = 1 - fpr, y = tpr)) +
    geom_line(color = "blue", size = 1) +
    geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
    annotate("text", x = 0.5, y = 0.2, label = paste("AUC =", round(metrics_svm$auc, 4)), size = 5) +
    labs(title = "Curva ROC para el modelo SVM",
         x = "1 - Especificidad",
         y = "Sensibilidad") +
    theme_minimal()
  roc_plot3_interactive <- ggplotly(roc_plot3)
  roc_plot3_interactive
})
  # Métricas de modelos predictivos
  output$metrics_rf_output <- renderPrint({
    metrics_rf
  })
  
  output$metrics_tree_output <- renderPrint({
    metrics_tree
  })
  
  output$metrics_svm_output <- renderPrint({
    metrics_svm
  })
  
  # Pruebas estadísticas
  test_results <- realizar_pruebas_estadisticas(data, 
                                                categorical_vars = c("Gender", "Smoking_Status", "Diabetic", "Dominant_Hand", "Family_History", "APOE_ε4", "Physical_Activity", "Depression_Status", "Medication_History", "Nutrition_Diet", "Sleep_Quality", "Chronic_Health_Conditions", "Education_Level"),
                                                numeric_vars = c("AlcoholLevel", "Age", "Heart_Rate", "BloodOxygenLevel", "BodyTemperature", "Weight", "MRI_Delay"))
  
  output$chi_square_output <- renderPrint({
    test_results$chi_square
  })
  
  output$t_test_output <- renderPrint({
    test_results$t_test
  })
  
  output$mann_whitney_output <- renderPrint({
    test_results$mann_whitney
  })
}
# Iniciar la aplicación Shiny
shinyApp(ui, server)

