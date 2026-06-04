import os
import argparse
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    cohen_kappa_score,
    log_loss
)


# ==========================================
# ARGUMENT PARSER (MLflow Project Params)
# ==========================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--n_estimators",
    type=int,
    default=100
)

parser.add_argument(
    "--max_depth",
    type=int,
    default=6
)

parser.add_argument(
    "--learning_rate",
    type=float,
    default=0.1
)

args = parser.parse_args()


# # ==========================================
# # DAGSHUB & MLFLOW CONFIGURATION
# # ==========================================

# os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv(
#     "MLFLOW_TRACKING_USERNAME", "muhammadrosid77"
# )

# os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv(
#     "MLFLOW_TRACKING_PASSWORD", "da2970cf9b321338c0f70f48816627787dde5122"
# )

# MLFLOW_TRACKING_URI = (
#     "https://dagshub.com/muhammadrosid77/"
#     "obesity_clasification.mlflow"
# )

# mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# EXPERIMENT_NAME = "Obesity_Classification_CI"

# mlflow.set_experiment(EXPERIMENT_NAME)


# ==========================================
# LOAD DATA
# ==========================================

def load_data():

    print("=" * 50)
    print("LOADING DATASET")
    print("=" * 50)

    df = pd.read_csv(
        "ObesityDataSet_preprocessing/"
        "ObesityDataSet_processed.csv"
    )

    X = df.drop(columns=["NObeyesdad"])
    y = df["NObeyesdad"]

    print(f"Dataset Shape  : {df.shape}")
    print(f"Features       : {X.shape[1]}")
    print(f"Target Classes : {y.nunique()}")

    return X, y


# ==========================================
# SPLIT DATA
# ==========================================

def split_data(X, y):

    print("\n" + "=" * 50)
    print("SPLITTING DATA")
    print("=" * 50)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"Train Size : {X_train.shape[0]}")
    print(f"Test Size  : {X_test.shape[0]}")

    return X_train, X_test, y_train, y_test


# ==========================================
# EVALUATE MODEL
# ==========================================

def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    metrics = {
        "accuracy": accuracy_score(
            y_test, y_pred
        ),
        "f1_weighted": f1_score(
            y_test, y_pred, average="weighted"
        ),
        "precision_weighted": precision_score(
            y_test, y_pred, average="weighted"
        ),
        "recall_weighted": recall_score(
            y_test, y_pred, average="weighted"
        ),
        "cohen_kappa": cohen_kappa_score(
            y_test, y_pred
        ),
        "log_loss": log_loss(
            y_test, y_pred_proba
        )
    }

    report = classification_report(
        y_test, y_pred
    )

    cm = confusion_matrix(y_test, y_pred)

    return metrics, report, cm, y_pred


# ==========================================
# MAIN
# ==========================================

def main():

    # Load & Split
    X, y = load_data()

    X_train, X_test, y_train, y_test = split_data(
        X, y
    )

    # ------------------------------------------
    # Model: XGBoost (Best from Kriteria 2)
    # ------------------------------------------

    print("\n" + "=" * 50)
    print("TRAINING: XGBoost (CI Pipeline)")
    print("=" * 50)

    params = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "subsample": 0.9,
        "colsample_bytree": 0.7,
        "min_child_weight": 1,
        "gamma": 0.1,
        "random_state": 42,
        "eval_metric": "mlogloss"
    }

    print(f"\nParameters: {params}")

    model = XGBClassifier(**params)

    with mlflow.start_run(
        run_name="CI_XGBoost"
    ) as run:

        # ----------------------------------
        # Log Parameters
        # ----------------------------------

        mlflow.log_param("model_name", "XGBoost")

        for key, value in params.items():
            mlflow.log_param(key, value)

        # ----------------------------------
        # Train Model
        # ----------------------------------

        model.fit(X_train, y_train)

        # ----------------------------------
        # Evaluate
        # ----------------------------------

        metrics, report, cm, y_pred = evaluate_model(
            model, X_test, y_test
        )

        print(f"\nAccuracy           : {metrics['accuracy']:.4f}")
        print(f"F1 (weighted)      : {metrics['f1_weighted']:.4f}")
        print(f"Precision (weighted): {metrics['precision_weighted']:.4f}")
        print(f"Recall (weighted)  : {metrics['recall_weighted']:.4f}")
        print(f"Cohen Kappa        : {metrics['cohen_kappa']:.4f}")
        print(f"Log Loss           : {metrics['log_loss']:.4f}")

        print(f"\nClassification Report:\n{report}")

        # ----------------------------------
        # Log Metrics
        # ----------------------------------

        for key, value in metrics.items():
            mlflow.log_metric(key, value)

        # ----------------------------------
        # Artifact 1: Classification Report
        # ----------------------------------

        report_path = "classification_report.txt"

        with open(report_path, "w") as f:
            f.write(report)

        mlflow.log_artifact(report_path)
        os.remove(report_path)

        # ----------------------------------
        # Artifact 2: Confusion Matrix
        # ----------------------------------

        fig, ax = plt.subplots(figsize=(8, 6))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=sorted(y_test.unique()),
            yticklabels=sorted(y_test.unique()),
            ax=ax
        )

        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix - XGBoost CI")

        cm_path = "confusion_matrix.png"
        fig.savefig(cm_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        mlflow.log_artifact(cm_path)
        os.remove(cm_path)

        # ----------------------------------
        # Artifact 3: Feature Importance
        # ----------------------------------

        importances = model.feature_importances_
        feature_names = X_train.columns

        feat_imp = pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        }).sort_values(
            "importance", ascending=False
        ).head(15)

        fig, ax = plt.subplots(figsize=(10, 6))

        sns.barplot(
            data=feat_imp,
            x="importance",
            y="feature",
            hue="feature",
            palette="viridis",
            legend=False,
            ax=ax
        )

        ax.set_title("Top 15 Feature Importance - XGBoost CI")
        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")

        fi_path = "feature_importance.png"
        fig.savefig(fi_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        mlflow.log_artifact(fi_path)
        os.remove(fi_path)

        # ----------------------------------
        # Log Model to MLflow
        # ----------------------------------

        mlflow.sklearn.log_model(
            model,
            artifact_path="model"
        )

        # ----------------------------------
        # Save Model Locally (for Docker)
        # ----------------------------------

        model_output_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "model_output"
        )

        mlflow.sklearn.save_model(
            model,
            model_output_path
        )

        # ----------------------------------
        # Save Run ID
        # ----------------------------------

        run_id = run.info.run_id

        run_id_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "run_id.txt"
        )

        with open(run_id_path, "w") as f:
            f.write(run_id)

        print(f"\nRun ID    : {run_id}")
        print(f"Model saved to: {model_output_path}")

        print(f"\n[SUCCESS] XGBoost CI logged to MLflow!")

    print("\n" + "=" * 50)
    print("CI TRAINING COMPLETED")
    print("=" * 50)


if __name__ == "__main__":
    main()
