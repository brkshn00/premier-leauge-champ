"""
main.py
Premier League RandomForest şampiyonluk tahmini ana çalıştırma dosyası.

Çalıştırma:
    python main.py

Çıktılar:
    output/tables/   -> CSV sonuç tabloları
    output/figures/  -> tüm grafikler
"""

from __future__ import annotations

from pathlib import Path
import warnings
import sys
import os

import pandas as pd

from feature_eng import build_dataset
from randomforest import (
    BASE_RF_PARAMS,
    evaluate_parameter_effects,
    extract_feature_importance,
    predict_future_champion,
    run_grid_search,
    walk_forward_evaluate,
    evaluate_fixed_holdout,
)
from visualize import (
    plot_feature_importance,
    plot_future_predictions,
    plot_grid_search_results,
    plot_numeric_distributions,
    plot_parameter_effects,
    plot_target_distribution,
)

warnings.filterwarnings("ignore")

ROOT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT_DIR / "data" / "premier_league_data_v2.csv"
OUTPUT_DIR = ROOT_DIR / "output"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"


def save_summary_metrics(name: str, result, table_dir: Path) -> None:
    metrics = pd.DataFrame(
        [
            {
                "model": name,
                "top1_hit_rate": result.top1_hit_rate,
                "top3_hit_rate": result.top3_hit_rate,
                "roc_auc": result.roc_auc,
                "average_precision": result.average_precision,
                "brier_score": result.brier_score,
            }
        ]
    )
    metrics.to_csv(table_dir / f"{name}_metrics.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    print("\n=== Premier League RandomForest Şampiyonluk Tahmini ===\n")
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    print("1) Veri okunuyor ve feature engineering yapılıyor...")
    full_df, train_df, future_df = build_dataset(DATA_PATH)
    full_df.to_csv(TABLE_DIR / "engineered_full_dataset.csv", index=False, encoding="utf-8-sig")
    train_df.to_csv(TABLE_DIR / "train_dataset.csv", index=False, encoding="utf-8-sig")
    future_df.to_csv(TABLE_DIR / "future_prediction_dataset.csv", index=False, encoding="utf-8-sig")

    latest_season = future_df["Sezon"].iloc[0]
    latest_year = int(future_df["Sezon_Yili"].iloc[0])
    print(f"   Son veri sezonu: {latest_season}")
    print(f"   Bu proje son veri sezonuna göre {latest_year + 1}-{latest_year + 2} sezonu için tahmin üretir.")

    print("\n2) Veri dağılımı grafikleri oluşturuluyor...")
    plot_target_distribution(train_df, FIGURE_DIR)
    plot_numeric_distributions(train_df, FIGURE_DIR)

    print("\n3) Base RandomForest walk-forward performansı ölçülüyor...")
    base_result = walk_forward_evaluate(train_df, BASE_RF_PARAMS)
    save_summary_metrics("base_randomforest", base_result, TABLE_DIR)
    base_result.detail.to_csv(TABLE_DIR / "base_walk_forward_predictions.csv", index=False, encoding="utf-8-sig")
    print(f"   Top-1 hit rate      : %{base_result.top1_hit_rate * 100:.2f}")
    print(f"   Top-3 hit rate      : %{base_result.top3_hit_rate * 100:.2f}")
    print(f"   ROC-AUC             : {base_result.roc_auc:.3f}")
    print(f"   Average precision   : {base_result.average_precision:.3f}")
    print(f"   Brier score         : {base_result.brier_score:.3f}")

    print("\n4) Tek tek parametre etkileri ölçülüyor...")
    parameter_results = evaluate_parameter_effects(train_df)
    parameter_results.to_csv(TABLE_DIR / "parameter_effect_results.csv", index=False, encoding="utf-8-sig")
    plot_parameter_effects(parameter_results, FIGURE_DIR / "parameter_effects")
    print("   Parametre etki grafikleri output/figures/parameter_effects içine kaydedildi.")

    print("\n5) GridSearchCV ile en iyi parametreler aranıyor...")
    grid, grid_results = run_grid_search(train_df)
    grid_results.to_csv(TABLE_DIR / "grid_search_results.csv", index=False, encoding="utf-8-sig")
    plot_grid_search_results(grid_results, FIGURE_DIR)
    print("   En iyi GridSearch parametreleri:")
    for key, value in grid.best_params_.items():
        print(f"   - {key}: {value}")
    print(f"   En iyi CV average precision: {grid.best_score_:.3f}")

    print("\n6) En iyi parametrelerle son 3 sezon holdout performansı ölçülüyor...")
    best_clean_params = {k.replace("model__", ""): v for k, v in grid.best_params_.items()}
    best_result = evaluate_fixed_holdout(train_df, best_clean_params)
    save_summary_metrics("gridsearch_best_randomforest", best_result, TABLE_DIR)
    best_result.detail.to_csv(TABLE_DIR / "gridsearch_best_holdout_predictions.csv", index=False, encoding="utf-8-sig")
    print(f"   Top-1 hit rate      : %{best_result.top1_hit_rate * 100:.2f}")
    print(f"   Top-3 hit rate      : %{best_result.top3_hit_rate * 100:.2f}")
    print(f"   ROC-AUC             : {best_result.roc_auc:.3f}")
    print(f"   Average precision   : {best_result.average_precision:.3f}")
    print(f"   Brier score         : {best_result.brier_score:.3f}")

    print("\n7) Gelecek sezon şampiyon tahmini yapılıyor...")
    future_predictions, final_model = predict_future_champion(train_df, future_df, grid.best_params_)
    future_predictions.to_csv(TABLE_DIR / "future_champion_predictions.csv", index=False, encoding="utf-8-sig")
    plot_future_predictions(future_predictions, FIGURE_DIR)

    predicted_champion = future_predictions.iloc[0]
    print(f"   Tahmini şampiyon: {predicted_champion['Takim']} ")
    print(f"   Olasılık skoru  : {predicted_champion['Champion_Probability']:.4f}")

    print("\n8) Feature importance çıkarılıyor...")
    importance_df = extract_feature_importance(final_model, top_n=25)
    importance_df.to_csv(TABLE_DIR / "feature_importance.csv", index=False, encoding="utf-8-sig")
    plot_feature_importance(importance_df, FIGURE_DIR)

    print("\nTamamlandı.")
    print(f"Grafikler: {FIGURE_DIR}")
    print(f"Tablolar : {TABLE_DIR}\n")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
