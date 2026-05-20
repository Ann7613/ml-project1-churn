from preprocessing import load_and_preprocess

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

from sklearn.ensemble import RandomForestClassifier


from sklearn.model_selection import GridSearchCV, StratifiedKFold

import pandas as pd


# cargar datos
X_train, X_val, y_train, y_val, X_test = (
    load_and_preprocess()
)

# modelo
log_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

# entrenamiento
log_model.fit(
    X_train,
    y_train
)

# predicciones
y_pred_log = log_model.predict(X_val)

log_acc = accuracy_score(
    y_val,
    y_pred_log
)

# métricas principales
log_f1 = f1_score(
    y_val,
    y_pred_log
)

# métricas
print("\nLOGISTIC REGRESSION\n")

print("Accuracy:")
print(
    accuracy_score(y_val, y_pred_log)
)

print("\nClassification Report:\n")
print(
    classification_report(
        y_val,
        y_pred_log
    )
)

print("\nConfusion Matrix:\n")
print(
    confusion_matrix(
        y_val,
        y_pred_log
    )
)



# =========================
# RANDOM FOREST
# =========================

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(
    X_train,
    y_train
)

y_pred_rf = rf_model.predict(X_val)

rf_acc = accuracy_score(
    y_val,
    y_pred_rf
)

rf_f1 = f1_score(
    y_val,
    y_pred_rf
)

print("\nRANDOM FOREST\n")

print("Accuracy:")
print(
    accuracy_score(y_val, y_pred_rf)
)

print("\nClassification Report:\n")
print(
    classification_report(
        y_val,
        y_pred_rf
    )
)

print("\nConfusion Matrix:\n")
print(
    confusion_matrix(
        y_val,
        y_pred_rf
    )
)



# =========================
# GridSearchCV 
# =========================


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# --- Tuning Logistic Regression ---
param_grid_log = {
    "C": [0.01, 0.1, 1, 10, 100],
    "solver": ["lbfgs", "liblinear"],
    "class_weight": [None, "balanced"],
}
gs_log = GridSearchCV(
    LogisticRegression(max_iter=1000, random_state=42),
    param_grid_log,
    cv=cv,
    scoring="f1",
    n_jobs=-1,
)
gs_log.fit(X_train, y_train)
log_model = gs_log.best_estimator_
print("Mejores params LR:", gs_log.best_params_)

# --- Tuning Random Forest ---
param_grid_rf = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_leaf": [1, 5, 10],
    "class_weight": [None, "balanced"],
}
gs_rf = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid_rf,
    cv=cv,
    scoring="f1",
    n_jobs=-1,
)
gs_rf.fit(X_train, y_train)
rf_model = gs_rf.best_estimator_
print("Mejores params RF:", gs_rf.best_params_)

# =========================
# TUNED LOGISTIC REGRESSION
# =========================

y_pred_log_tuned = log_model.predict(X_val)

log_tuned_acc = accuracy_score(
    y_val,
    y_pred_log_tuned
)

log_tuned_f1 = f1_score(
    y_val,
    y_pred_log_tuned
)

print("\nTUNED LOGISTIC REGRESSION\n")

print("Accuracy:")
print(log_tuned_acc)

print("\nF1 Score:")
print(log_tuned_f1)


# =========================
# TUNED RANDOM FOREST
# =========================

y_pred_rf_tuned = rf_model.predict(X_val)

rf_tuned_acc = accuracy_score(
    y_val,
    y_pred_rf_tuned
)

rf_tuned_f1 = f1_score(
    y_val,
    y_pred_rf_tuned
)

print("\nTUNED RANDOM FOREST\n")

print("Accuracy:")
print(rf_tuned_acc)

print("\nF1 Score:")
print(rf_tuned_f1)


print("\n========================")
print("MODEL COMPARISON")
print("========================")

print("\nLogistic Regression")
print("Accuracy:", log_acc)
print("F1:", log_f1)

print("\nRandom Forest")
print("Accuracy:", rf_acc)
print("F1:", rf_f1)

print("\nTuned Logistic Regression")
print("Accuracy:", log_tuned_acc)
print("F1:", log_tuned_f1)

print("\nTuned Random Forest")
print("Accuracy:", rf_tuned_acc)
print("F1:", rf_tuned_f1)



# =========================
# TABLA COMPARATIVA
# =========================


comparison_df = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest",
        "Tuned Logistic Regression",
        "Tuned Random Forest"
    ],
    "Accuracy": [
        log_acc,
        rf_acc,
        log_tuned_acc,
        rf_tuned_acc
    ],
    "F1 Score": [
        log_f1,
        rf_f1,
        log_tuned_f1,
        rf_tuned_f1
    ]
})

print("\nCOMPARISON TABLE\n")
print(comparison_df)
