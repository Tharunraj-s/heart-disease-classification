# -------------------------------
# Load Required Libraries
# -------------------------------
library(ggplot2)
library(dplyr)
library(readr)
library(caret)
library(xgboost)
library(e1071)
library(pROC)
library(randomForest)

# -------------------------------
# Load & Preprocess Data
# -------------------------------
heart_data <- read_csv("HeartDiseaseData.csv") %>%
  mutate(across(c(Smoking, AlcoholDrinking, Stroke, Sex, AgeCategory, HeartDisease), as.factor)) %>%
  na.omit()

# -------------------------------
# Train-Test Split
# -------------------------------
set.seed(42)
train_index <- createDataPartition(heart_data$HeartDisease, p = 0.7, list = FALSE)
train_data <- heart_data[train_index, ]
test_data <- heart_data[-train_index, ]

# -------------------------------
# Model 1: XGBoost with caret (CV + VarImp)
# -------------------------------
ctrl <- trainControl(method = "cv", classProbs = TRUE, summaryFunction = twoClassSummary)

train_data_caret <- train_data %>%
  mutate(HeartDisease = relevel(HeartDisease, ref = "No"))

quiet_train <- function(...) {
  sink(tempfile())
  on.exit(sink())
  invisible(train(...))
}

set.seed(42)
xgb_caret <- quiet_train(
  HeartDisease ~ ., data = train_data_caret,
  method = "xgbTree",
  trControl = ctrl,
  metric = "ROC"
)

# Predictions and Performance: XGBoost
xgb_probs_caret <- predict(xgb_caret, newdata = test_data, type = "prob")[, "Yes"]
roc_xgb_caret <- roc(test_data$HeartDisease, xgb_probs_caret)
plot(roc_xgb_caret, main = "ROC Curve - XGBoost (caret-tuned)")
cat("AUC (XGBoost caret):", auc(roc_xgb_caret), "\n")

xgb_preds <- predict(xgb_caret, newdata = test_data)
xgb_cm <- confusionMatrix(xgb_preds, test_data$HeartDisease, positive = "Yes")
print(xgb_cm)

print(xgb_caret)
plot(xgb_caret, main = "XGBoost - CV ROC by Hyperparameter")

xgb_imp <- varImp(xgb_caret)
plot(xgb_imp, top = 10, main = "XGBoost - Top 10 Important Features (caret)")

# -------------------------------
# Model 2: Random Forest with caret
# -------------------------------
set.seed(42)
rf_caret <- train(HeartDisease ~ ., data = train_data_caret,
                  method = "rf",
                  metric = "ROC",
                  trControl = ctrl,
                  tuneLength = 10,
                  ntree = 500,
                  tuneGrid = expand.grid(mtry = c(2, 4, 6)))

# Predictions and Performance: Random Forest
rf_probs <- predict(rf_caret, newdata = test_data, type = "prob")[, "Yes"]
roc_rf <- roc(test_data$HeartDisease, rf_probs)
plot(roc_rf, main = "ROC Curve - Random Forest (Tuned)")
cat("AUC (RF tuned):", auc(roc_rf), "\n")

rf_preds <- predict(rf_caret, newdata = test_data)
rf_cm <- confusionMatrix(rf_preds, test_data$HeartDisease, positive = "Yes")
print(rf_cm)

print(rf_caret)
plot(rf_caret, main = "Random Forest - ROC by mtry")

rf_imp <- varImp(rf_caret)
plot(rf_imp, top = 10, main = "Random Forest - Top 10 Important Features")


# XGBoost F1
xgb_precision <- xgb_cm$byClass["Precision"]
xgb_recall <- xgb_cm$byClass["Sensitivity"]
xgb_f1 <- 2 * ((xgb_precision * xgb_recall) / (xgb_precision + xgb_recall))
cat("F1 Score (XGBoost):", round(xgb_f1, 4), "\n")

# Random Forest F1
rf_precision <- rf_cm$byClass["Precision"]
rf_recall <- rf_cm$byClass["Sensitivity"]
rf_f1 <- 2 * ((rf_precision * rf_recall) / (rf_precision + rf_recall))
cat("F1 Score (Random Forest):", round(rf_f1, 4), "\n")

# View best mtry and plot performance
xgb_caret$results %>%
  arrange(desc(ROC)) %>%
  head(5)
