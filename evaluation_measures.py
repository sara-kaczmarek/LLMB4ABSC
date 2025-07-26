
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import string
import re
from tqdm import tqdm
import unicodedata

def compute_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def compute_macro_precision(y_true, y_pred):
    return precision_score(y_true, y_pred, average="macro", zero_division=0)

def compute_macro_recall(y_true, y_pred):
    return recall_score(y_true, y_pred, average="macro", zero_division=0)

def compute_macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro", zero_division=0)

def evaluate_predictions(df, true_col="true_sentiment", pred_col="predicted_sentiment"):
    y_true = df[true_col].str.lower()
    y_pred = df[pred_col].str.lower()

    # Keep only valid predictions
    valid_mask = y_pred.isin(["positive", "neutral", "negative"])
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]

    results = {
        "Accuracy": compute_accuracy(y_true, y_pred),
        "Macro-Precision": compute_macro_precision(y_true, y_pred),
        "Macro-Recall": compute_macro_recall(y_true, y_pred),
        "Macro-F1": compute_macro_f1(y_true, y_pred),
    }

    return results

def print_confusion_matrix(df, true_col="true_sentiment", pred_col="predicted_sentiment"):
    y_true = df[true_col].str.lower()
    y_pred = df[pred_col].str.lower()

    # Filter only valid predictions
    valid_classes = ["positive", "neutral", "negative"]
    mask = y_pred.isin(valid_classes)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    cm = confusion_matrix(y_true, y_pred, labels=valid_classes)
    cm_df = pd.DataFrame(cm, index=[f"True {label}" for label in valid_classes],
                            columns=[f"Pred {label}" for label in valid_classes])

    print(cm_df)
    return cm_df

def create_results_df(name, metrics):
    metrics = {key: round(value, 4) for key, value in metrics.items()}
    df = pd.DataFrame.from_dict(metrics, orient='index', columns=[name])
    return df

def create_results_column(df_results, metrics, k, retrieval_df_name):
    retrieval_df_name = retrieval_df_name.lower()
    col_name = f"{k}-Shot_{retrieval_df_name.capitalize()}"
    metrics = {key: round(value, 4) for key, value in metrics.items()}
    df = pd.DataFrame.from_dict(metrics, orient='index', columns=[col_name])
    return df
