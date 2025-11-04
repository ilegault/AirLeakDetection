"""
Hyperparameter optimization for model tuning.

Provides grid search, random search, and Bayesian optimization via Optuna.
"""

import numpy as np
from typing import Dict, Any, Callable, List, Tuple, Optional
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import warnings

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


class GridSearchTuner:
    """Exhaustive grid search hyperparameter tuning.
    
    Args:
        param_grid: Dictionary of parameter names to lists of values
        cv: Number of cross-validation folds
        scoring: Scoring metric
        n_jobs: Number of parallel jobs
    """
    
    def __init__(
        self,
        param_grid: Dict[str, List[Any]],
        cv: int = 5,
        scoring: str = "accuracy",
        n_jobs: int = -1
    ):
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.results = None
        self.best_params = None
        self.best_score = None
    
    def search(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray
    ) -> dict:
        """
        Perform grid search.
        
        Args:
            model: Scikit-learn compatible model
            X: Feature array
            y: Label array
            
        Returns:
            Dictionary with search results
        """
        grid_search = GridSearchCV(
            model,
            self.param_grid,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            verbose=1
        )
        
        grid_search.fit(X, y)
        
        self.best_params = grid_search.best_params_
        self.best_score = grid_search.best_score_
        
        # Extract results
        results = {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "cv_results": grid_search.cv_results_,
            "param_grid": self.param_grid,
            "method": "grid_search"
        }
        
        return results
    
    def get_results_dataframe(self):
        """Get results as pandas DataFrame."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas required for dataframe results")
        
        if self.results is None:
            return None
        
        return pd.DataFrame(self.results["cv_results"])


class RandomSearchTuner:
    """Random search hyperparameter tuning.
    
    Args:
        param_distributions: Dictionary of parameter distributions
        n_iter: Number of parameter combinations to sample
        cv: Number of cross-validation folds
        scoring: Scoring metric
        n_jobs: Number of parallel jobs
        random_state: Random seed
    """
    
    def __init__(
        self,
        param_distributions: Dict[str, Any],
        n_iter: int = 20,
        cv: int = 5,
        scoring: str = "accuracy",
        n_jobs: int = -1,
        random_state: int = 42
    ):
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.best_params = None
        self.best_score = None
    
    def search(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray
    ) -> dict:
        """
        Perform random search.
        
        Args:
            model: Scikit-learn compatible model
            X: Feature array
            y: Label array
            
        Returns:
            Dictionary with search results
        """
        random_search = RandomizedSearchCV(
            model,
            self.param_distributions,
            n_iter=self.n_iter,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=1
        )
        
        random_search.fit(X, y)
        
        self.best_params = random_search.best_params_
        self.best_score = random_search.best_score_
        
        results = {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "cv_results": random_search.cv_results_,
            "param_distributions": self.param_distributions,
            "n_iter": self.n_iter,
            "method": "random_search"
        }
        
        return results


class BayesianOptimizationTuner:
    """Bayesian optimization using Optuna.
    
    Args:
        objective_fn: Function to minimize/maximize
        param_space: Dictionary defining parameter space
        n_trials: Number of trials
        direction: 'minimize' or 'maximize'
        n_jobs: Number of parallel jobs
    """
    
    def __init__(
        self,
        objective_fn: Callable,
        param_space: Dict[str, Any],
        n_trials: int = 100,
        direction: str = "maximize",
        n_jobs: int = 1
    ):
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna required for Bayesian optimization. "
                            "Install with: pip install optuna")
        
        self.objective_fn = objective_fn
        self.param_space = param_space
        self.n_trials = n_trials
        self.direction = direction
        self.n_jobs = n_jobs
        self.study = None
        self.best_params = None
        self.best_value = None
    
    def search(self, **kwargs) -> dict:
        """
        Perform Bayesian optimization.
        
        Args:
            **kwargs: Additional arguments passed to objective function
            
        Returns:
            Dictionary with search results
        """
        def objective(trial):
            # Suggest parameters
            params = self._suggest_params(trial)
            
            # Evaluate
            score = self.objective_fn(trial, params, **kwargs)
            return score
        
        self.study = optuna.create_study(direction=self.direction)
        
        self.study.optimize(
            objective,
            n_trials=self.n_trials,
            n_jobs=self.n_jobs,
            show_progress_bar=True
        )
        
        self.best_params = self.study.best_params
        self.best_value = self.study.best_value
        
        results = {
            "best_params": self.best_params,
            "best_value": self.best_value,
            "n_trials": self.n_trials,
            "trials": [
                {
                    "trial_id": trial.number,
                    "value": trial.value,
                    "params": trial.params
                }
                for trial in self.study.trials
            ],
            "method": "bayesian_optimization"
        }
        
        return results
    
    def _suggest_params(self, trial) -> dict:
        """Suggest parameters based on parameter space."""
        params = {}
        
        for param_name, param_config in self.param_space.items():
            param_type = param_config.get("type", "float")
            
            if param_type == "float":
                params[param_name] = trial.suggest_float(
                    param_name,
                    param_config["low"],
                    param_config["high"],
                    log=param_config.get("log", False)
                )
            elif param_type == "int":
                params[param_name] = trial.suggest_int(
                    param_name,
                    param_config["low"],
                    param_config["high"]
                )
            elif param_type == "categorical":
                params[param_name] = trial.suggest_categorical(
                    param_name,
                    param_config["choices"]
                )
        
        return params
    
    def get_trials_dataframe(self):
        """Get trials as pandas DataFrame."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas required for dataframe")
        
        if self.study is None:
            return None
        
        return self.study.trials_dataframe()


class HyperparameterTuner:
    """Unified hyperparameter tuning interface.
    
    Args:
        method: Tuning method ('grid', 'random', 'bayesian')
        **kwargs: Method-specific arguments
    """
    
    def __init__(self, method: str = "grid", **kwargs):
        self.method = method
        self.kwargs = kwargs
        self.tuner = None
        self.best_params = None
        self.best_score = None
    
    def create_tuner(self):
        """Create tuner instance."""
        if self.method == "grid":
            self.tuner = GridSearchTuner(**self.kwargs)
        elif self.method == "random":
            self.tuner = RandomSearchTuner(**self.kwargs)
        elif self.method == "bayesian":
            self.tuner = BayesianOptimizationTuner(**self.kwargs)
        else:
            raise ValueError(f"Unknown tuning method: {self.method}")
    
    def search(self, model, X: np.ndarray, y: np.ndarray) -> dict:
        """Perform hyperparameter search."""
        if self.tuner is None:
            self.create_tuner()
        
        results = self.tuner.search(model, X, y)
        self.best_params = results.get("best_params")
        self.best_score = results.get("best_score") or results.get("best_value")
        
        return results
    
    def get_best_params(self) -> dict:
        """Get best parameters found."""
        return self.best_params
    
    def get_best_score(self) -> float:
        """Get best score found."""
        return self.best_score


def suggest_hyperparameter_space(
    model_type: str
) -> Dict[str, List[Any]]:
    """
    Suggest hyperparameter search space for a model type.
    
    Args:
        model_type: Type of model ('rf', 'svm', 'xgb', 'lr')
        
    Returns:
        Parameter grid for grid search
    """
    if model_type == "rf":
        return {
            "n_estimators": [50, 100, 200],
            "max_depth": [5, 10, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        }
    
    elif model_type == "svm":
        return {
            "C": [0.1, 1, 10, 100],
            "kernel": ["linear", "rbf", "poly"],
            "gamma": ["scale", "auto"]
        }
    
    elif model_type == "xgb":
        return {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1, 0.3],
            "subsample": [0.6, 0.8, 1.0]
        }
    
    elif model_type == "lr":
        return {
            "C": [0.001, 0.01, 0.1, 1.0],
            "penalty": ["l1", "l2"],
            "solver": ["liblinear"]
        }
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")