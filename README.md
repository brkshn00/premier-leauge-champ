# Premier League RandomForest Şampiyonluk Tahmini

Bu proje, Premier League sezon verileriyle `RandomForestClassifier` kullanarak bir sonraki sezonun şampiyonunu tahmin eder.

## Dosyalar

- `feature_eng.py`: veri okuma, hedef değişken oluşturma, feature engineering
- `randomforest.py`: RandomForest modeli, walk-forward validasyon, parametre etkisi, GridSearch
- `visualize.py`: veri dağılımı, parametre etkisi, GridSearch, feature importance ve tahmin grafikleri
- `main.py`: tüm akışı çalıştıran ana dosya

## Çalıştırma

```bash
pip install -r requirements.txt
python main.py
```

## Üretilen çıktılar

- `output/figures/data_target_distribution.png`
- `output/figures/data_numeric_distributions.png`
- `output/figures/parameter_effects/parameter_effect_*.png`
- `output/figures/grid_search_best_combinations.png`
- `output/figures/feature_importance.png`
- `output/figures/future_champion_predictions.png`
- `output/tables/*.csv`

## Not

Veri seti 2020-2021 sezonunda bittiği için modelin "gelecek sezon" tahmini 2021-2022 içindir. 2026-2027 gibi güncel bir sezon tahmini için veri setinin güncellenmesi gerekir.
