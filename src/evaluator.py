import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from src.models import svm, knn, decision_tree, random_forest, xgboost, lightgbm, catboost
from src.dataset import exoplanet_dataset
from utils.loader import load_env

ENV = load_env()


def evaluate():
    dataset = exoplanet_dataset(root=ENV["exoplanet_dataset_path"], train=True)

    models = {
        "svm": svm(),
        "knn": knn(),
        "decision_tree": decision_tree(),
        "random_forest": random_forest(),
        "xgboost": xgboost(),
        "lightgbm": lightgbm(),
        "catboost": catboost()
    }

    for model_name, model in models.items():
        print(f"\n--- Evaluating {model_name} model ---")
        path = os.path.join("weights/ml", model_name)
        model.load(path)
        results = model.evaluate(dataset.data, dataset.labels)
        print(f"Results for {model_name}:")
        print('Accuracy', results["accuracy"])
        print('Precision', results["precision"])
        print('Recall', results["recall"])
        print('F1-Score', results["f1-score"])

if __name__ == "__main__":
    evaluate()