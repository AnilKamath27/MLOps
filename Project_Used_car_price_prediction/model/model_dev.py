import logging
from abc import ABC, abstractmethod
import optuna
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

class Model(ABC):
    """Abstract base class for all models."""
    @abstractmethod
    def train(self, x_train, y_train,):
        """Trains the model on the given data.
        Args:
            x_train: Training data
            y_train: Target data"""
        pass

    @abstractmethod
    def optimize(self, trial, x_train, y_train, x_test, y_test):
        """Optimizes the hyperparameters of the model.
        Args:
            trial: Optuna trial object
            x_train: Training data
            y_train: Target data
            x_test: Testing data
            y_test: Testing target"""
        pass

class RandomForestRegressorModel(Model):
    """RandomForestModel that implements the Model interface."""
    def train(self, x_train, y_train, **kwargs):
        reg = RandomForestRegressor(random_state=42, n_jobs=-1, **kwargs)
        reg.fit(x_train, y_train)
        return reg

    def optimize(self, trial, x_train, y_train, x_test, y_test):
        logging.info("Model is being optimized")
        n_estimators = trial.suggest_int("n_estimators", 100, 1000)
        max_depth = trial.suggest_int("max_depth", 3, 10)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        reg = self.train(x_train, y_train, n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split)
        return reg.score(x_test, y_test)

class GradientBoostingRegressorModel(Model):
    """GradientBoostingRegressorModel that implements the Model interface."""
    def train(self, x_train, y_train, **kwargs):
        reg = GradientBoostingRegressor(random_state=42, **kwargs)
        reg.fit(x_train, y_train)
        return reg

    def optimize(self, trial, x_train, y_train, x_test, y_test):
            logging.info("Model is being optimized")
            n_estimators = trial.suggest_int("n_estimators", 100, 1000)
            max_depth = trial.suggest_int("max_depth", 3, 10)
            learning_rate = trial.suggest_loguniform("learning_rate", 0.001, 0.1)
            subsample = trial.suggest_uniform("subsample", 0.5, 1.0)
            min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
            alpha_val = trial.suggest_loguniform("alpha", 0.001, 1.0)
            reg = self.train(x_train, y_train, n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
                    subsample=subsample,min_samples_split=min_samples_split, alpha=alpha_val)
            return reg.score(x_test, y_test)

class XGBoostRegressorModel(Model):
    """XGBoostModel that implements the Model interface."""
    def train(self, x_train, y_train, **kwargs):
        reg = XGBRegressor(n_estimators=786,max_depth=3,learning_rate=0.055739208637784324,subsample=0.7563441283737317,colsample_bytree=0.6343152551882135,
                           min_child_weight=3, reg_lambda=0.08378662808828764,alpha =0.018293844833644084,n_jobs=-1, random_state=42, **kwargs)
        reg.fit(x_train, y_train)
        return reg

    def optimize(self, trial, x_train, y_train, x_test, y_test):
        logging.info("Model is being optimized")
        n_estimators = trial.suggest_int("n_estimators", 100, 1000)
        max_depth = trial.suggest_int("max_depth", 3, 10)
        learning_rate = trial.suggest_loguniform("learning_rate", 0.001, 0.1)
        subsample = trial.suggest_uniform("subsample", 0.5, 1.0)
        colsample_bytree = trial.suggest_uniform("colsample_bytree", 0.5, 1.0)
        min_child_weight = trial.suggest_int("min_child_weight", 1, 10)
        lambda_val = trial.suggest_loguniform("lambda", 1e-3, 10.0)
        alpha_val = trial.suggest_loguniform("alpha", 1e-3, 10.0)
        reg = self.train(x_train, y_train, n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, subsample=subsample, 
                        colsample_bytree=colsample_bytree, min_child_weight=min_child_weight, alpha=alpha_val,reg_lambda=lambda_val)
        return reg.score(x_test, y_test)

class LinearRegressionModel(Model):
    """LinearRegressionModel that implements the Model interface."""
    def train(self, x_train, y_train, **kwargs):
        reg = LinearRegression(**kwargs)
        reg.fit(x_train, y_train)
        logging.info("Model training is completed")
        return reg

    # For linear regression, there might not be hyperparameters that we want to tune, so we can simply return the score
    def optimize(self, trial, x_train, y_train, x_test, y_test):
        reg = self.train(x_train, y_train)
        return reg.score(x_test, y_test)

class HyperparameterTuner:
    """Class for performing hyperparameter tuning. It uses Model strategy to perform tuning."""
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def optimize(self, n_trials=100):
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: self.model.optimize(trial, self.x_train, self.y_train, self.x_test, self.y_test), n_trials=n_trials)
        return study.best_trial.params
