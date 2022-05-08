from dataclasses import dataclass

import joblib

import mlflow as mlflow
from sklearn.datasets import load_iris
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split


@dataclass
class _Constants:
    MLFLOW_SERVER_URL: str = 'http://127.0.0.1:5000/'
    PARAMS_GB = {'n_estimators': [100, 200, 500, 800, 1200],
                 'max_depth': [10, 20, 30, 40, 50, 60, 70, None],
                 'min_samples_split': [1, 2, 3, 5],
                 'min_samples_leaf': [1, 2, 3, 5],
                 'max_features': ['auto', 'sqrt'],
                 'learning_rate': [0.1, 0.2, 0.5, 0.8]}


if __name__ == '__main__':
    iris = load_iris()

    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2018)

    with mlflow.start_run(run_name='iris'):
        model = GradientBoostingClassifier()
        clf = RandomizedSearchCV(model, param_distributions=_Constants.PARAMS_GB)
        clf.fit(X_train, y_train)

        predictions = clf.predict(X_test)
        mlflow.sklearn.log_model(clf, 'gbc_model')
        [mlflow.log_param(param, value) for param, value in _Constants.PARAMS_GB.items()]

        # metrics
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        print("  mse: {}".format(mse))
        print("  mae: {}".format(mae))
        print("  R2: {}".format(r2))

        mlflow.log_metric("mse", mse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        # saving results
        joblib.dump(clf, 'gbc_model.mdl')
