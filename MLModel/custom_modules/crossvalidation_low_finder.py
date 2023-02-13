from xgboost import XGBClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, precision_score
from sklearn.ensemble import BaggingClassifier


def objective_rf(trial, X_train, y_train ):

    rf_n_estimators = trial.suggest_int("n_estimators", 100, 500, 50)
    rf_max_depth = trial.suggest_int("max_depth", 2, 20, 2)
    rf_min_samples_split = trial.suggest_int("min_samples_split", 2, 10, 2)
    
    model = RandomForestClassifier(
        n_estimators = rf_n_estimators,
        max_depth = rf_max_depth,
        min_samples_split = rf_min_samples_split
    )

    score = cross_val_score(model, X_train, y_train, cv=3, scoring='roc_auc', n_jobs = -1)
    error = score.mean()
    
    return error

def objective_adb(trial, X_train, y_train ):

    adb_n_estimators = trial.suggest_int("n_estimators", 20, 200, 30)
    adb_learning_rate = trial.suggest_float('learning_rate',  1e-6, 3)
    
    
    model = AdaBoostClassifier(
        n_estimators = adb_n_estimators,
        learning_rate = adb_learning_rate
    )

    score = cross_val_score(model, X_train, y_train, cv=3, scoring='roc_auc', n_jobs = -1)
    error = score.mean()
    
    return error

def objective_xgb(trial, X_train, y_train ):

    xgb_n_estimators = trial.suggest_int("n_estimators", 50, 500, 50)
    xgb_learning_rate = trial.suggest_float('learning_rate',  1e-6, 2)
    xgb_max_depth = trial.suggest_int('max_depth',  3, 15)
    
    model = XGBClassifier(
        n_estimators = xgb_n_estimators,
        learning_rate = xgb_learning_rate,
        max_depth = xgb_max_depth,
    )

    score = cross_val_score(model, X_train, y_train, cv=3, scoring='roc_auc', n_jobs = -1)
    error = score.mean()
    
    return error