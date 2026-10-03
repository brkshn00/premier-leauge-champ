"""
visualize.py
Model ve veri çıktıları için grafik fonksiyonları.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from feature_eng import NUMERIC_FEATURES, TARGET


def _ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_current_figure(path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def plot_target_distribution(train_df: pd.DataFrame, output_dir: str | Path) -> Path:
    """Şampiyon / şampiyon değil hedef dağılımı."""
    output_dir = _ensure_dir(output_dir)
    counts = train_df[TARGET].value_counts().sort_index()
    labels = ["Şampiyon değil", "Gelecek sezon şampiyon"]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, [counts.get(0, 0), counts.get(1, 0)])
    plt.title("Hedef Değişken Dağılımı")
    plt.ylabel("Satır sayısı")
    plt.xticks(rotation=15)

    path = output_dir / "data_target_distribution.png"
    _save_current_figure(path)
    return path


def plot_numeric_distributions(train_df: pd.DataFrame, output_dir: str | Path) -> Path:
    """Önemli sayısal değişkenlerin histogramlarını tek görselde gösterir."""
    output_dir = _ensure_dir(output_dir)
    selected = [
        "Son_Sezon_Sira",
        "Gol_Farki",
        "PPG_Puan_Ortalamasi",
        "Yas_Ortalamasi",
        "Piyasa_Degeri_Mn",
        "Transfer_Harcamasi",
        "Kadro_Genisligi",
        "Sampiyonluk_Son5",
    ]
    selected = [c for c in selected if c in train_df.columns]

    n_cols = 2
    n_rows = int(np.ceil(len(selected) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    axes = np.asarray(axes).reshape(-1)

    for ax, col in zip(axes, selected):
        ax.hist(train_df[col].dropna(), bins=20)
        ax.set_title(f"{col} dağılımı")
        ax.set_xlabel(col)
        ax.set_ylabel("Frekans")

    for ax in axes[len(selected):]:
        ax.axis("off")

    path = output_dir / "data_numeric_distributions.png"
    _save_current_figure(path)
    return path


def plot_parameter_effects(parameter_results: pd.DataFrame, output_dir: str | Path) -> list[Path]:
    """Her RandomForest parametresi için ayrı etki grafiği oluşturur."""
    output_dir = _ensure_dir(output_dir)
    created: list[Path] = []

    for param in parameter_results["parameter"].dropna().unique():
        sub = parameter_results[(parameter_results["parameter"] == param) & (parameter_results["status"] == "ok")].copy()
        if sub.empty:
            continue

        x = np.arange(len(sub))
        labels = sub["value"].astype(str).tolist()

        plt.figure(figsize=(10, 5))
        plt.plot(x, sub["top1_hit_rate"], marker="o", label="Top-1 hit rate")
        plt.plot(x, sub["top3_hit_rate"], marker="o", label="Top-3 hit rate")
        plt.plot(x, sub["average_precision"], marker="o", label="Average precision")
        plt.title(f"RandomForest parametre etkisi: {param}")
        plt.xlabel(param)
        plt.ylabel("Skor")
        plt.xticks(x, labels, rotation=30, ha="right")
        plt.ylim(0, 1.05)
        plt.legend()
        plt.grid(True, alpha=0.3)

        path = output_dir / f"parameter_effect_{param}.png"
        _save_current_figure(path)
        created.append(path)

    return created


def plot_grid_search_results(grid_results: pd.DataFrame, output_dir: str | Path, top_n: int = 12) -> Path:
    """GridSearch'teki en iyi kombinasyonları bar grafikte gösterir."""
    output_dir = _ensure_dir(output_dir)
    top = grid_results.sort_values("rank_test_score").head(top_n).copy()
    top = top.iloc[::-1]

    labels = []
    for _, row in top.iterrows():
        label = (
            f"est={row.get('param_model__n_estimators')} | "
            f"depth={row.get('param_model__max_depth')} | "
            f"leaf={row.get('param_model__min_samples_leaf')} | "
            f"feat={row.get('param_model__max_features')} | "
            f"cw={row.get('param_model__class_weight')}"
        )
        labels.append(label)

    plt.figure(figsize=(13, 7))
    plt.barh(labels, top["mean_test_score"])
    plt.title("GridSearch En İyi Parametre Kombinasyonları")
    plt.xlabel("Mean CV Average Precision")
    plt.ylabel("Parametre kombinasyonu")

    path = output_dir / "grid_search_best_combinations.png"
    _save_current_figure(path)
    return path


def plot_feature_importance(feature_importance: pd.DataFrame, output_dir: str | Path) -> Path:
    """RandomForest feature importance grafiği."""
    output_dir = _ensure_dir(output_dir)
    data = feature_importance.sort_values("importance", ascending=True)

    plt.figure(figsize=(10, max(6, len(data) * 0.35)))
    plt.barh(data["feature"], data["importance"])
    plt.title("RandomForest Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")

    path = output_dir / "feature_importance.png"
    _save_current_figure(path)
    return path


def plot_future_predictions(prediction_df: pd.DataFrame, output_dir: str | Path, top_n: int = 10) -> Path:
    """Gelecek sezon şampiyonluk olasılığı en yüksek adaylar."""
    output_dir = _ensure_dir(output_dir)
    top = prediction_df.head(top_n).sort_values("Champion_Probability", ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(top["Takim"], top["Champion_Probability"])
    plt.title(f"Gelecek Sezon Şampiyonluk Tahmini - İlk {top_n}")
    plt.xlabel("Tahmini şampiyonluk olasılığı")
    plt.ylabel("Takım")

    path = output_dir / "future_champion_predictions.png"
    _save_current_figure(path)
    return path
