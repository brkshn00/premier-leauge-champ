"""
feature_eng.py
Premier League şampiyonluk tahmini için veri okuma ve feature engineering.

Mantık:
- Her satır bir takımın bir sezondaki durumudur.
- Hedef değişken: Bu takım GELECEK sezonda şampiyon oldu mu?
- Son sezonun hedefi bilinmediği için prediction/future set olarak ayrılır.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


NUMERIC_FEATURES: List[str] = [
    "Son_Sezon_Sira",
    "Gol_Farki",
    "PPG_Puan_Ortalamasi",
    "Onceki_Sezon_Sira",
    "Avrupa_Kupasi",
    "Yas_Ortalamasi",
    "Piyasa_Degeri_Mn",
    "Kadro_Genisligi",
    "Transfer_Harcamasi",
    "PPG_Onceki",
    "PPG_Trend",
    "PPG_Son3",
    "Sira_Gecen",
    "Sira_Trend",
    "Top4_Son3",
    "Sampiyonluk_Son5",
    "Piyasa_Log",
    "Piyasa_Buyume",
    "Transfer_Log",
]

CATEGORICAL_FEATURES: List[str] = ["Takim"]
FEATURES: List[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "Next_Season_Champion"
SEASON_COL = "Sezon_Yili"


def load_data(csv_path: str | Path) -> pd.DataFrame:
    """CSV dosyasını okur, sezon yılını çıkarır ve temel kolonları temizler."""
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path, sep=";", decimal=",", encoding="latin-1")
    df = df.drop(columns=[c for c in df.columns if "Unnamed" in c], errors="ignore")

    market_cols = [c for c in df.columns if "Piyasa" in c]
    if market_cols and market_cols[0] != "Piyasa_Degeri_Mn":
        df = df.rename(columns={market_cols[0]: "Piyasa_Degeri_Mn"})

    df["Sezon_Yili"] = df["Sezon"].astype(str).str.split("-").str[0].astype(int)
    df = df.sort_values(["Takim", "Sezon_Yili"]).reset_index(drop=True)
    return df


def add_historical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takım bazlı geçmiş performans özellikleri üretir.
    shift(1) kullanıldığı için bu özellikler sadece geçmiş sezonlardan gelir.
    """
    df = df.copy()
    group = df.groupby("Takim", group_keys=False)

    df["PPG_Onceki"] = group["PPG_Puan_Ortalamasi"].shift(1)
    df["PPG_Trend"] = group["PPG_Puan_Ortalamasi"].shift(1) - group["PPG_Puan_Ortalamasi"].shift(2)
    df["PPG_Son3"] = group["PPG_Puan_Ortalamasi"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean()
    )

    df["Sira_Gecen"] = group["Son_Sezon_Sira"].shift(1)
    df["Sira_Trend"] = group["Son_Sezon_Sira"].shift(2) - group["Son_Sezon_Sira"].shift(1)
    df["Top4_Son3"] = group["Son_Sezon_Sira"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).apply(lambda x: np.sum(x <= 4), raw=True)
    )
    df["Sampiyonluk_Son5"] = group["is_champion"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).sum()
    )

    df["Piyasa_Log"] = np.log1p(df["Piyasa_Degeri_Mn"].clip(lower=0))
    df["Piyasa_Buyume"] = group["Piyasa_Degeri_Mn"].transform(lambda s: s.shift(1).pct_change(fill_method=None))
    df["Transfer_Log"] = np.log1p(df["Transfer_Harcamasi"].clip(lower=0))

    return df


def add_next_season_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hedef değişkeni oluşturur:
    Bir takımın bulunduğu sezondan sonraki sezonda şampiyon olup olmadığı.

    Önemli düzeltme:
    groupby shift(-1) doğrudan kullanılmaz; çünkü takım ligden düşüp yıllar sonra dönebilir.
    Bu yüzden takım + bir sonraki takvim sezonu üzerinden merge yapılır.
    """
    df = df.copy()
    next_df = df[["Takim", "Sezon_Yili", "is_champion"]].copy()
    next_df["Sezon_Yili"] = next_df["Sezon_Yili"] - 1
    next_df = next_df.rename(columns={"is_champion": TARGET})

    df = df.merge(next_df, on=["Takim", "Sezon_Yili"], how="left")

    max_year = df["Sezon_Yili"].max()
    df["is_future_row"] = df["Sezon_Yili"].eq(max_year)

    # Son sezonun gerçek gelecek hedefi bilinmez. Eğitim dışı bırakacağız.
    df.loc[df["is_future_row"], TARGET] = np.nan

    # Önceki sezonlarda takım bir sonraki sezon ligde yoksa şampiyon olamaz: 0.
    df.loc[~df["is_future_row"], TARGET] = df.loc[~df["is_future_row"], TARGET].fillna(0)

    return df


def build_dataset(csv_path: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns
    -------
    full_df : tüm feature'ları içeren dataframe
    train_df: hedefi bilinen satırlar
    future_df: son sezon satırları, hedefi bilinmeyen prediction adayları
    """
    df = load_data(csv_path)
    df = add_historical_features(df)
    df = add_next_season_target(df)

    df[NUMERIC_FEATURES] = df[NUMERIC_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
    df[CATEGORICAL_FEATURES] = df[CATEGORICAL_FEATURES].fillna("Unknown")

    train_df = df[~df["is_future_row"]].copy()
    train_df[TARGET] = train_df[TARGET].astype(int)

    future_df = df[df["is_future_row"]].copy()

    return df, train_df, future_df
