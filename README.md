# Heart Disease Classification

Academic statistical-learning project by Tharunraj Singaravelu, comparing XGBoost and Random Forest using R and caret.

## Workflow

Categorical preprocessing, incomplete-row removal, stratified 70/30 split, cross-validation, ROC-AUC, confusion matrices and feature importance. Test confusion matrices explicitly treat `Yes` as the positive class.

## Run

Provide the original coursework input `HeartDiseaseData.csv` in the repository root. Expected columns are `HeartDisease`, `BMI`, `Smoking`, `AlcoholDrinking`, `Stroke`, `PhysicalHealth`, `MentalHealth`, `Sex`, `AgeCategory`, `SleepTime`. The report describes it as a BRFSS-derived subset; its exact extraction source and redistribution terms have not been established, so no dataset is included. Public BRFSS information: https://www.cdc.gov/brfss/

```r
install.packages(c("ggplot2", "dplyr", "readr", "caret", "xgboost", "e1071", "pROC", "randomForest"))
source("code.R")
```

## Evaluation and release notes

The release uses a neutral random seed of 42 instead of the original identifier-based seed. Results will differ from saved coursework outputs. No historical accuracy, sensitivity or F1 claims are carried over. Re-run with the same data and record package versions before reporting metrics. Cross-validation uses caret's first class (`No`) for its two-class summary, while final confusion matrices use `Yes`; interpret the class-specific measures accordingly.

This is an educational analysis, not a clinical screening tool. R execution has not been verified in this release environment. Reports, PDFs, student identifiers and original history remain in the private archive.
