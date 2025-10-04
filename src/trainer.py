import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from src.models import svm, knn, decision_tree, random_forest, xgboost, lightgbm, catboost

def train():
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
        print(f"\n--- Training {model_name} model ---")
        model.train(data)
        path = os.path.join("weights/ml", model_name)
        dir_path = os.path.dirname(path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        model.save(path)

