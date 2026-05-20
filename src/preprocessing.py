import pandas as pd

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from sklearn.model_selection import train_test_split


def load_and_preprocess():

    # cargar datasets
    train_df = pd.read_csv("../train.csv")
    test_df = pd.read_csv("../test.csv")

    # convertir TotalCharges
    train_df["TotalCharges"] = pd.to_numeric(
        train_df["TotalCharges"],
        errors="coerce"
    )

    test_df["TotalCharges"] = pd.to_numeric(
        test_df["TotalCharges"],
        errors="coerce"
    )

    # eliminar customerID
    train_df = train_df.drop("customerID", axis=1)
    test_df = test_df.drop("customerID", axis=1)

    # separar variables
    X = train_df.drop("Churn", axis=1)

    y = train_df["Churn"].map({
        "No": 0,
        "Yes": 1
    })

    # rellenar nulls
    median_total = X["TotalCharges"].median()

    X["TotalCharges"] = X["TotalCharges"].fillna(
        median_total
    )

    test_df["TotalCharges"] = test_df[
        "TotalCharges"
    ].fillna(median_total)

    # columnas categóricas
    categorical_columns = X.select_dtypes(
        include=["object", "string"]
    ).columns

    # one hot encoding
    encoder = OneHotEncoder(
        drop="first",
        sparse_output=False
    )

    encoded_train = encoder.fit_transform(
        X[categorical_columns]
    )

    encoded_test = encoder.transform(
        test_df[categorical_columns]
    )

    encoded_train_df = pd.DataFrame(
        encoded_train,
        columns=encoder.get_feature_names_out(
            categorical_columns
        )
    )

    encoded_test_df = pd.DataFrame(
        encoded_test,
        columns=encoder.get_feature_names_out(
            categorical_columns
        )
    )

    # columnas numéricas
    numeric_columns = X.select_dtypes(
        include=["int64", "float64"]
    ).columns

    numeric_train_df = X[numeric_columns].reset_index(
        drop=True
    )

    numeric_test_df = test_df[
        numeric_columns
    ].reset_index(drop=True)

    # unir todo
    final_X = pd.concat(
        [numeric_train_df, encoded_train_df],
        axis=1
    )

    final_test = pd.concat(
        [numeric_test_df, encoded_test_df],
        axis=1
    )

    # split
    X_train, X_val, y_train, y_val = train_test_split(
        final_X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # scaling
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)

    X_val = scaler.transform(X_val)

    final_test = scaler.transform(final_test)

    return (
        X_train,
        X_val,
        y_train,
        y_val,
        final_test
    )
