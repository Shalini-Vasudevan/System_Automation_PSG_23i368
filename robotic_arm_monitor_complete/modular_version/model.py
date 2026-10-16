"""Requirement 2 - predictive wear detection (ML).

Logistic regression on [torque %, temperature, vibration, hours since service].
Two models are trained:
  * full model       - uses all four features
  * no-torque model  - used in 'degraded mode' when the torque sensor is
                       suspected of being spoofed (so we don't trust it).
"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from config import RATED_TORQUE_NM

FULL = ["torque_pct", "temp_c", "vibration_mm_s", "hours_since_service"]
NO_TORQUE = ["temp_c", "vibration_mm_s", "hours_since_service"]


def load_history(path="historical_joint_data.csv"):
    df = pd.read_csv(path)
    df["torque_pct"] = df.apply(
        lambda r: r.torque_nm / RATED_TORQUE_NM[r.joint] * 100.0, axis=1)
    return df


def _pipe():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


class WearModel:
    def __init__(self, df):
        y = df["failed_within_30d"]
        self.full = _pipe().fit(df[FULL], y)
        self.no_torque = _pipe().fit(df[NO_TORQUE], y)
        # honest accuracy estimates (5-fold cross-validation)
        self.cv_full = cross_val_score(_pipe(), df[FULL], y, cv=5).mean()
        self.cv_no_torque = cross_val_score(_pipe(), df[NO_TORQUE], y, cv=5).mean()
        Xtr, Xte, ytr, yte = train_test_split(df[FULL], y, test_size=0.25,
                                              random_state=0, stratify=y)
        self.holdout_acc = _pipe().fit(Xtr, ytr).score(Xte, yte)

    def predict(self, torque_pct, temp_c, vib, hours, torque_trusted=True):
        """Probability that this joint fails within 30 days."""
        if torque_trusted:
            X = pd.DataFrame([[torque_pct, temp_c, vib, hours]], columns=FULL)
            return float(self.full.predict_proba(X)[0, 1])
        X = pd.DataFrame([[temp_c, vib, hours]], columns=NO_TORQUE)
        return float(self.no_torque.predict_proba(X)[0, 1])
