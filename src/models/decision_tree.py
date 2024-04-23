# -*- coding: utf-8 -*-
""" 
Vanilla Decision Tree (not important at the moment, cf. notebook for final models)
"""
import click
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from src.helpers import read_csv

class DecisionTreeModel:
    """ Decision tree """
    def __init__(self, data, target_variable, embeddings):
        self.decision_tree = DecisionTreeRegressor()

        self.train_data = read_csv(data["train"])
        self.test_data = read_csv(data["test"])

        self.X_train = np.load(embeddings["train"])
        self.X_test = np.load(embeddings["test"])
        self.y_train = self.train_data[target_variable].values.tolist()
        self.y_test = self.test_data[target_variable].values.tolist()

        self.metrics = [
            ("mae", mean_absolute_error),
            ("mse", mean_squared_error),
            ("r2", r2_score)]
    
    def get_metrics(self, y_true, y_pred):
        res = {}
        for (label, metric) in self.metrics:
            res[label] = metric(y_true, y_pred)
        return res

    
    def fit(self):
        self.decision_tree.fit(self.X_train, self.y_train)
        y_pred = self.predict(self.X_train)
        metrics = self.get_metrics(y_true=self.y_train, y_pred=y_pred)
        print(metrics)
    
    def predict(self, X_input):
        return self.decision_tree.predict(X_input)
    
    def evaluate(self):
        y_pred = self.predict(self.X_test)
        metrics = self.get_metrics(y_true=self.y_test, y_pred=y_pred)
        print(metrics)


@click.command()
@click.option("--train_path", help="Path to training dataset")
@click.option("--test_path", help="Path to test dataset")
@click.option("--train_embeddings", help="Path to training embeddings")
@click.option("--test_embeddings", help="Path to test embeddings")
@click.option("--target", help="Target variable to predict")
def main(train_path, test_path, train_embeddings, test_embeddings, target):
    """ Train decision tree """
    decision_tree = DecisionTreeModel(
        data={"train": train_path, "test": test_path},
        target_variable=target,
        embeddings={"train": train_embeddings, "test": test_embeddings}
    )
    print("Training model and retrieving metrics on train set")
    decision_tree.fit()
    print("Evaluating model on test set")
    decision_tree.evaluate()


if __name__ == '__main__':
    main()
    