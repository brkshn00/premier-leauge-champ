"""
randomforest.py
RandomForestClassifier modeli, walk-forward değerlendirme, parametre etkisi analizi
ve GridSearch işlemleri.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from feature_eng import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, SEASON_COL, TARGET


BASE_RF_PARAMS: Dict[str, Any] = {
    "n_estimators": 80,
    "max_depth": 6,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "bootstrap": True,
    "criterion": "gini",
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": 1,
}

# Tek tek parametre etkisi grafikleri için değerler.
# Her denemede sadece ilgili parametre değişir, diğerleri BASE_RF_PARAMS olarak sabit kalır.
PARAMETER_TEST_VALUES: Dict[str, List[Any]] = {
    "n_estimators": [30, 80, 150],
    "max_depth": [None, 6],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt", "log2"],
    "bootstrap": [True, False],
    "criterion": ["gini", "entropy"],
    "class_weight": [None, "balanced"],
}

# GridSearch için daha kontrollü bir aralık: proje hızlı çalışsın ama yine de tüm istenen
# parametre aileleri birlikte optimize edilsin.
GRID_SEARCH_PARAMS: Dict[str, List[Any]] = {
    "model__n_estimators": [50],
    "model__max_depth": [None, 6],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf": [1],
    "model__max_features": ["sqrt"],
    "model__bootstrap": [True],
    "model__criterion": ["gini", "entropy"],
    "model__class_weight": ["balanced"],
}


@dataclass
class WalkForwardResult:
    top1_hit_rate: float
    top3_hit_rate: float
    roc_auc: float
    average_precision: float
    brier_score: float
    detail: pd.DataFrame


def _make_one_hot_encoder() -> OneHotEncoder:
    """Farklı sklearn sürümlerinde uyumlu OneHotEncoder üretir."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # Eski sklearn sürümleri için
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_pipeline(rf_params: Dict[str, Any] | None = None) -> Pipeline:
    """Preprocess + RandomForestClassifier pipeline'ı oluşturur."""
    params = BASE_RF_PARAMS.copy()
    if rf_params:
        params.update(rf_params)

    numeric_pipe = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _make_one_hot_encoder()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    model = RandomForestClassifier(**params)
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def get_X_y(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Model input-output ayırımı."""
    return df[FEATURES + [SEASON_COL]].copy(), df[TARGET].astype(int).copy()


def _model_features_only(X: pd.DataFrame) -> pd.DataFrame:
    """Pipeline'a sadece model feature'larını verir; Sezon_Yili CV için tutulur ama modele girmez."""
    return X[FEATURES].copy()


def build_walk_forward_cv(df: pd.DataFrame, min_train_seasons: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Zaman sırasını bozmayan CV splitleri üretir.
    Her test sezonu için önceki tüm sezonlar eğitimdir.
    """
    seasons = sorted(df[SEASON_COL].unique())
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    index_array = np.arange(len(df))

    for test_season in seasons[min_train_seasons:]:
        train_mask = df[SEASON_COL].to_numpy() < test_season
        test_mask = df[SEASON_COL].to_numpy() == test_season
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue
        splits.append((index_array[train_mask], index_array[test_mask]))

    return splits


def build_recent_holdout_split(df: pd.DataFrame, holdout_seasons: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parametre karşılaştırması ve GridSearch için hızlı, zamansal holdout spliti.
    Son birkaç sezon test, önceki sezonlar eğitim olur.
    """
    seasons = sorted(df[SEASON_COL].unique())
    if len(seasons) <= holdout_seasons:
        raise ValueError("Holdout split için yeterli sezon yok.")

    test_seasons = set(seasons[-holdout_seasons:])
    index_array = np.arange(len(df))
    train_mask = ~df[SEASON_COL].isin(test_seasons).to_numpy()
    test_mask = df[SEASON_COL].isin(test_seasons).to_numpy()
    return index_array[train_mask], index_array[test_mask]


def evaluate_fixed_holdout(df: pd.DataFrame, rf_params: Dict[str, Any] | None = None, holdout_seasons: int = 3) -> WalkForwardResult:
    """
    Hızlı parametre deneyi için son N sezonu test olarak ayırır.
    Tek parametre etkisi grafikleri bu fonksiyonu kullanır; böylece her değer için diğerleri sabit kalır.
    """
    X_all, y_all = get_X_y(df)
    train_idx, test_idx = build_recent_holdout_split(df, holdout_seasons=holdout_seasons)

    model = make_pipeline(rf_params)
    model.fit(_model_features_only(X_all.iloc[train_idx]), y_all.iloc[train_idx])

    test_df = df.iloc[test_idx].copy()
    proba = model.predict_proba(_model_features_only(X_all.iloc[test_idx]))[:, 1]
    test_df["Champion_Probability"] = proba

    rows: List[Dict[str, Any]] = []
    top1_hits = 0
    top3_hits = 0
    evaluated_seasons = 0

    for season_year, season_df in test_df.groupby(SEASON_COL):
        season_df = season_df.sort_values("Champion_Probability", ascending=False)
        predicted_champion = season_df.iloc[0]["Takim"]
        top3_teams = season_df.head(3)["Takim"].tolist()
        actual_champions = season_df[season_df[TARGET] == 1]["Takim"].tolist()

        top1_hit = int(predicted_champion in actual_champions)
        top3_hit = int(any(team in actual_champions for team in top3_teams))

        top1_hits += top1_hit
        top3_hits += top3_hit
        evaluated_seasons += 1

        rows.append({
            "input_season": f"{int(season_year)}-{int(season_year) + 1}",
            "predicted_for_season": f"{int(season_year) + 1}-{int(season_year) + 2}",
            "predicted_champion": predicted_champion,
            "actual_champion": ", ".join(actual_champions) if actual_champions else "Unknown/Not in candidate set",
            "top3_candidates": ", ".join(top3_teams),
            "top1_hit": top1_hit,
            "top3_hit": top3_hit,
        })

    y_test = y_all.iloc[test_idx].to_numpy()
    try:
        roc_auc = float(roc_auc_score(y_test, proba))
    except ValueError:
        roc_auc = np.nan

    return WalkForwardResult(
        top1_hit_rate=top1_hits / evaluated_seasons,
        top3_hit_rate=top3_hits / evaluated_seasons,
        roc_auc=roc_auc,
        average_precision=float(average_precision_score(y_test, proba)),
        brier_score=float(brier_score_loss(y_test, proba)),
        detail=pd.DataFrame(rows),
    )


def walk_forward_evaluate(df: pd.DataFrame, rf_params: Dict[str, Any] | None = None) -> WalkForwardResult:
    """
    Sezon bazlı gerçekçi değerlendirme.
    Her sezon için önceki sezonlarla eğit, o sezon takımlarından gelecek sezon şampiyonunu tahmin et.
    """
    rows: List[Dict[str, Any]] = []
    all_y: List[int] = []
    all_proba: List[float] = []

    splits = build_walk_forward_cv(df)
    X_all, y_all = get_X_y(df)

    top1_hits = 0
    top3_hits = 0
    evaluated_seasons = 0

    for train_idx, test_idx in splits:
        X_train = _model_features_only(X_all.iloc[train_idx])
        y_train = y_all.iloc[train_idx]
        X_test_full = X_all.iloc[test_idx].copy()
        X_test = _model_features_only(X_test_full)
        y_test = y_all.iloc[test_idx]

        model = make_pipeline(rf_params)
        model.fit(X_train, y_train)

        proba = model.predict_proba(X_test)[:, 1]
        test_df = df.iloc[test_idx].copy()
        test_df["Champion_Probability"] = proba
        test_df = test_df.sort_values("Champion_Probability", ascending=False)

        predicted_champion = test_df.iloc[0]["Takim"]
        top3_teams = test_df.head(3)["Takim"].tolist()
        actual_champions = test_df[test_df[TARGET] == 1]["Takim"].tolist()

        top1_hit = int(predicted_champion in actual_champions)
        top3_hit = int(any(team in actual_champions for team in top3_teams))

        top1_hits += top1_hit
        top3_hits += top3_hit
        evaluated_seasons += 1

        all_y.extend(y_test.tolist())
        all_proba.extend(proba.tolist())

        season_year = int(test_df.iloc[0][SEASON_COL])
        rows.append(
            {
                "input_season": f"{season_year}-{season_year + 1}",
                "predicted_for_season": f"{season_year + 1}-{season_year + 2}",
                "predicted_champion": predicted_champion,
                "actual_champion": ", ".join(actual_champions) if actual_champions else "Unknown/Not in candidate set",
                "top3_candidates": ", ".join(top3_teams),
                "top1_hit": top1_hit,
                "top3_hit": top3_hit,
            }
        )

    if evaluated_seasons == 0:
        raise ValueError("Walk-forward değerlendirme için yeterli sezon yok.")

    all_y_arr = np.asarray(all_y)
    all_p_arr = np.asarray(all_proba)

    try:
        roc_auc = float(roc_auc_score(all_y_arr, all_p_arr))
    except ValueError:
        roc_auc = np.nan

    avg_precision = float(average_precision_score(all_y_arr, all_p_arr))
    brier = float(brier_score_loss(all_y_arr, all_p_arr))

    return WalkForwardResult(
        top1_hit_rate=top1_hits / evaluated_seasons,
        top3_hit_rate=top3_hits / evaluated_seasons,
        roc_auc=roc_auc,
        average_precision=avg_precision,
        brier_score=brier,
        detail=pd.DataFrame(rows),
    )


def evaluate_parameter_effects(df: pd.DataFrame) -> pd.DataFrame:
    """
    RandomForest parametrelerinin etkisini ölçer.
    Bir parametre denenirken diğer tüm parametreler sabit tutulur.
    """
    records: List[Dict[str, Any]] = []

    for param_name, values in PARAMETER_TEST_VALUES.items():
        for value in values:
            test_params = {param_name: value}
            try:
                result = evaluate_fixed_holdout(df, test_params)
                records.append(
                    {
                        "parameter": param_name,
                        "value": str(value),
                        "raw_value": value,
                        "top1_hit_rate": result.top1_hit_rate,
                        "top3_hit_rate": result.top3_hit_rate,
                        "roc_auc": result.roc_auc,
                        "average_precision": result.average_precision,
                        "brier_score": result.brier_score,
                        "status": "ok",
                    }
                )
            except Exception as exc:  # Değer bazı sklearn sürümlerinde desteklenmeyebilir.
                records.append(
                    {
                        "parameter": param_name,
                        "value": str(value),
                        "raw_value": value,
                        "top1_hit_rate": np.nan,
                        "top3_hit_rate": np.nan,
                        "roc_auc": np.nan,
                        "average_precision": np.nan,
                        "brier_score": np.nan,
                        "status": f"error: {exc}",
                    }
                )

    return pd.DataFrame(records)


def run_grid_search(df: pd.DataFrame) -> Tuple[GridSearchCV, pd.DataFrame]:
    """GridSearchCV ile en iyi RandomForest parametrelerini arar."""
    X, y = get_X_y(df)
    cv_splits = [build_recent_holdout_split(df, holdout_seasons=3)]

    pipeline = make_pipeline()
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=GRID_SEARCH_PARAMS,
        scoring="average_precision",
        cv=cv_splits,
        n_jobs=1,
        refit=True,
        verbose=0,
        return_train_score=True,
    )
    grid.fit(_model_features_only(X), y)

    results = pd.DataFrame(grid.cv_results_).sort_values("rank_test_score")
    return grid, results


def predict_future_champion(train_df: pd.DataFrame, future_df: pd.DataFrame, best_params: Dict[str, Any] | None = None) -> Tuple[pd.DataFrame, Pipeline]:
    """Son sezon verileriyle bir sonraki sezon şampiyon adaylarını olasılık sırasına dizer."""
    rf_params = BASE_RF_PARAMS.copy()
    if best_params:
        # GridSearch parametreleri model__ prefix'iyle gelir; onu temizliyoruz.
        cleaned = {k.replace("model__", ""): v for k, v in best_params.items()}
        rf_params.update(cleaned)

    model = make_pipeline(rf_params)

    X_train, y_train = get_X_y(train_df)
    model.fit(_model_features_only(X_train), y_train)

    future_X = future_df[FEATURES].copy()
    proba = model.predict_proba(future_X)[:, 1]

    prediction_df = future_df[["Sezon", "Takim", "Son_Sezon_Sira", "PPG_Puan_Ortalamasi", "Piyasa_Degeri_Mn"]].copy()
    prediction_df["Champion_Probability"] = proba
    prediction_df = prediction_df.sort_values("Champion_Probability", ascending=False).reset_index(drop=True)
    prediction_df["Predicted_Rank"] = np.arange(1, len(prediction_df) + 1)
    return prediction_df, model


def extract_feature_importance(model: Pipeline, top_n: int = 25) -> pd.DataFrame:
    """Eğitilmiş pipeline'dan feature importance tablosu çıkarır."""
    preprocessor = model.named_steps["preprocessor"]
    rf_model = model.named_steps["model"]

    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = np.asarray(FEATURES)

    importances = rf_model.feature_importances_
    importance_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    importance_df = importance_df.sort_values("importance", ascending=False).reset_index(drop=True)
    return importance_df.head(top_n)
