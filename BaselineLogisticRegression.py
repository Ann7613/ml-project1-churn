import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.evaluation import evaluate_model
from src.threshold import evaluate_thresholds

# cargar datasets
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# EDA 1: distribución de la variable objetivo Churn
plt.figure(figsize=(5, 4))
sns.countplot(data=train_df, x="Churn")
plt.title("Distribución de Churn")
plt.xlabel("Churn")
plt.ylabel("Cantidad")
plt.tight_layout()
plt.savefig("figures/churn_distribution.png", dpi=300)
plt.show()

# mostrar primeras filas
print("TRAIN:")
print(train_df.head())

print("\n====================\n")

print("TEST:")
print(test_df.head())

print("\n====================\n")

# informacion general
print(train_df.info())

print("\n====================\n")

# valores nulos
print("Valores nulos:")
print(train_df.isnull().sum())

# convertir TotalCharges a numero
train_df["TotalCharges"] = pd.to_numeric(
    train_df["TotalCharges"],
    errors="coerce"
)

test_df["TotalCharges"] = pd.to_numeric(
    test_df["TotalCharges"],
    errors="coerce"
)

# EDA 2: distribución de MonthlyCharges según Churn
plt.figure(figsize=(7, 5))
sns.histplot(
    data=train_df,
    x="MonthlyCharges",
    hue="Churn",
    kde=True
)
plt.title("MonthlyCharges según Churn")
plt.xlabel("Monthly Charges")
plt.ylabel("Frecuencia")
plt.tight_layout()
plt.savefig("figures/monthlycharges_churn.png", dpi=300)
plt.show()

print("\nTipos luego de conversion:\n")
print(train_df.dtypes)

print("\nValores nulos despues conversion:\n")
print(train_df.isnull().sum())

#se elimina el custumerID porque no lo necesitamos...

train_df = train_df.drop("customerID", axis=1)
test_df = test_df.drop("customerID", axis=1)

#separamos en churn...
X_train = train_df.drop("Churn", axis=1)

y_train = train_df["Churn"]

#lo convertimos a 1/0
y_train = y_train.map({
    "No": 0,
    "Yes": 1
})

#rellenamos los valores nulos de TotalCharges con la mediana...

median_total = X_train["TotalCharges"].median()

X_train["TotalCharges"] = X_train["TotalCharges"].fillna(median_total)

test_df["TotalCharges"] = test_df["TotalCharges"].fillna(median_total)


print("\nNulls en train:")
print(X_train.isnull().sum())

print("\nNulls en test:")
print(test_df.isnull().sum())

#convertiremos las variables categoricas con onehot encoding
categorical_columns = X_train.select_dtypes(
    include=["object", "string"]
).columns

print("\nColumnas categoricas:\n")
print(categorical_columns)

encoder = OneHotEncoder(
    drop="first",
    sparse_output=False
)


encoded_train = encoder.fit_transform(
    X_train[categorical_columns]
)

encoded_test = encoder.transform(
    test_df[categorical_columns]
)

print("\nShape encoded train:")
print(encoded_train.shape)

print("\nShape encoded test:")
print(encoded_test.shape)



encoded_train_df = pd.DataFrame(
    encoded_train,
    columns=encoder.get_feature_names_out(categorical_columns)
)

encoded_test_df = pd.DataFrame(
    encoded_test,
    columns=encoder.get_feature_names_out(categorical_columns)
)

#obtenemoss columnas numéricas

numeric_columns = X_train.select_dtypes(
    include=["int64", "float64"]
).columns

print("\nColumnas numericas:\n")
print(numeric_columns)

#separamos las numericas:


numeric_train_df = X_train[numeric_columns].reset_index(drop=True)

numeric_test_df = test_df[numeric_columns].reset_index(drop=True)


#ahora unimos todo y se hace un dataset entrenable...

final_X_train = pd.concat(
    [numeric_train_df, encoded_train_df],
    axis=1
)

final_X_test = pd.concat(
    [numeric_test_df, encoded_test_df],
    axis=1
)

# separar train y validation

X_train_split, X_val, y_train_split, y_val = train_test_split(
    final_X_train,
    y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train
)

# aplicar scaling correctamente

scaler = StandardScaler()

X_train_split = scaler.fit_transform(X_train_split)

X_val = scaler.transform(X_val)

X_test_scaled = scaler.transform(final_X_test)

# entrenar modelo

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(
    X_train_split,
    y_train_split
)

# predicciones

y_pred = model.predict(X_val)

# metricas

evaluate_model(
    y_val,
    y_pred,
    model_name="Logistic Regression Baseline"
)

#accuracy = accuracy_score(y_val, y_pred)

#print("\nAccuracy:")
#print(accuracy)

#print("\nClassification Report:\n")
#print(classification_report(y_val, y_pred))

#print("\nConfusion Matrix:\n")
#print(confusion_matrix(y_val, y_pred))

# probabilidades del modelo

y_prob = model.predict_proba(X_val)[:, 1]

# evaluar distintos thresholds

threshold_results = evaluate_thresholds(
    y_val,
    y_prob
)

print("\nThreshold tuning:\n")

print(threshold_results)

best_row = threshold_results.sort_values(
    by="f1",
    ascending=False
).iloc[0]

best_threshold = best_row["threshold"]

print("\nMejor threshold segun F1:")
print(best_row)

y_pred_best = (y_prob >= best_threshold).astype(int)

evaluate_model(
    y_val,
    y_pred_best,
    model_name=f"Logistic Regression Threshold {best_threshold}"
)

plt.figure(figsize=(7, 5))

plt.plot(
    threshold_results["threshold"],
    threshold_results["f1"],
    marker="o",
    label="F1-score"
)

plt.plot(
    threshold_results["threshold"],
    threshold_results["recall"],
    marker="o",
    label="Recall"
)

plt.plot(
    threshold_results["threshold"],
    threshold_results["precision"],
    marker="o",
    label="Precision"
)

plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Metricas segun Threshold")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/threshold_metrics.png", dpi=300)
plt.show()