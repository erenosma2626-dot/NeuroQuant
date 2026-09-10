import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, Any

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from neuro_modules.quant_benchmark import (
    fetch_asset_and_benchmark,
    compute_quant_features,
    QuantileLightGBMCluster
)
from backend.config import settings

TEST_DAYS = 126  # Son 6 ay (126 işlem günü kör veri)

def evaluate_ticker_model(ticker: str) -> Dict[str, Any]:
    benchmark = settings.get_benchmark(ticker)
    df_raw = fetch_asset_and_benchmark(ticker, benchmark, period="5y")
    df_feat = compute_quant_features(df_raw)
    
    # NaN temizliği ve Target kontrolü
    # target_5d son 5 günde NaN olacağı için onu dropna ile sadece test edilebilir günleri alıyoruz
    valid_mask = ~df_feat['target_5d'].isna()
    df_valid = df_feat.loc[valid_mask].copy()
    
    total_len = len(df_valid)
    train_len = max(200, total_len - TEST_DAYS)
    
    train_df = df_valid.iloc[:train_len].copy()
    test_df = df_valid.iloc[train_len:].copy()
    
    # 1. Modeli SADECE train verisiyle eğit (Zero Leakage)
    model = QuantileLightGBMCluster(ticker)
    model.fit(train_df, train_df['target_5d'])
    
    # 2. Kör test setinde tahmin al
    preds = model.predict_cone(test_df)
    
    actual_5d = test_df['target_5d'].values
    pred_5d = preds['median']
    lower_80 = preds['lower_80']
    upper_80 = preds['upper_80']
    
    # A. Yön Doğruluğu (Directional Accuracy / Win Rate)
    # Model yönü bildi mi? (+ ise +, - ise -)
    correct_dir = np.sign(pred_5d) == np.sign(actual_5d)
    directional_accuracy = np.mean(correct_dir) * 100.0
    
    # B. Long Doğruluğu (Model Yükseliş dediğinde gerçekten yükseldi mi?)
    long_mask = pred_5d > 0
    if np.sum(long_mask) > 0:
        long_precision = np.mean(actual_5d[long_mask] > 0) * 100.0
        long_count = int(np.sum(long_mask))
    else:
        long_precision = 0.0
        long_count = 0
        
    # C. Short Doğruluğu (Model Düşüş dediğinde gerçekten düştü mü?)
    short_mask = pred_5d < 0
    if np.sum(short_mask) > 0:
        short_precision = np.mean(actual_5d[short_mask] < 0) * 100.0
        short_count = int(np.sum(short_mask))
    else:
        short_precision = 0.0
        short_count = 0
        
    # D. Güven Konisi Kapsama Oranı (Coverage Rate - Hedef %80)
    inside_cone = (actual_5d >= lower_80) & (actual_5d <= upper_80)
    coverage_rate = np.mean(inside_cone) * 100.0
    
    # E. Sayısal Hata (MAE & RMSE)
    mae_pct = np.mean(np.abs(pred_5d - actual_5d)) * 100.0
    rmse_pct = np.sqrt(np.mean((pred_5d - actual_5d) ** 2)) * 100.0
    
    # F. 1 Yıllık Test (252 gün) için de hesaplayalım
    test_1y_len = min(252, total_len - 300)
    if test_1y_len > 150:
        train_1y = df_valid.iloc[:-test_1y_len].copy()
        test_1y = df_valid.iloc[-test_1y_len:].copy()
        m_1y = QuantileLightGBMCluster(ticker).fit(train_1y, train_1y['target_5d'])
        p_1y = m_1y.predict_cone(test_1y)
        act_1y = test_1y['target_5d'].values
        acc_1y = np.mean(np.sign(p_1y['median']) == np.sign(act_1y)) * 100.0
        cov_1y = np.mean((act_1y >= p_1y['lower_80']) & (act_1y <= p_1y['upper_80'])) * 100.0
    else:
        acc_1y = directional_accuracy
        cov_1y = coverage_rate

    return {
        "ticker": ticker,
        "test_samples": len(test_df),
        "directional_accuracy_6m": round(directional_accuracy, 2),
        "directional_accuracy_1y": round(acc_1y, 2),
        "long_precision": round(long_precision, 2),
        "long_count": long_count,
        "short_precision": round(short_precision, 2),
        "short_count": short_count,
        "coverage_80_pct": round(coverage_rate, 2),
        "coverage_1y_pct": round(cov_1y, 2),
        "mae_pct": round(mae_pct, 2),
        "rmse_pct": round(rmse_pct, 2),
    }

def main():
    test_universe = ["NVDA", "AAPL", "MSFT", "TSLA", "BTC-USD", "THYAO.IS"]
    print("=" * 70)
    print("🔬 NEUROQUANT 3.0: TAHMİN MODELİ ACCURACY & GÜVENİRLİK TESTİ")
    print("=" * 70)
    print(f"Metodoloji: Sıfır Sızıntı (Zero Data Leakage), 5-Günlük İleri Getiri (target_5d)")
    print(f"Test Dönemi: Son 6 Ay (126 iş günü) & Son 1 Yıl (252 iş günü) Kör Veri")
    print("-" * 70)
    
    results = []
    for t in test_universe:
        try:
            print(f"⏳ {t} verileri çekilip model test ediliyor...")
            res = evaluate_ticker_model(t)
            results.append(res)
            print(f"   ✓ {t}: 6-Ay Yön Başarısı: %{res['directional_accuracy_6m']} | 1-Yıl Yön Başarısı: %{res['directional_accuracy_1y']} | Kapsama: %{res['coverage_80_pct']} | MAE: %{res['mae_pct']}")
        except Exception as e:
            print(f"   ✗ Hata {t}: {e}")
            
    df_res = pd.DataFrame(results)
    print("\n" + "=" * 70)
    print("📊 ÖZET PERFORMANS TABLOSU")
    print("=" * 70)
    print(df_res.to_string(index=False))
    
    print("\n" + "-" * 70)
    print(f"🏆 GENEL EVREN ORTALAMALARI:")
    print(f"   • Ortalama 6-Aylık Yön Doğruluğu (Directional Accuracy): %{df_res['directional_accuracy_6m'].mean():.2f}")
    print(f"   • Ortalama 1-Yıllık Yön Doğruluğu (1-Year Directional):  %{df_res['directional_accuracy_1y'].mean():.2f}")
    print(f"   • Ortalama Yükseliş Başarısı (Long Precision):          %{df_res['long_precision'].mean():.2f}")
    print(f"   • Ortalama Düşüş Başarısı (Short Precision):           %{df_res['short_precision'].mean():.2f}")
    print(f"   • Güven Konisi Kapsama Oranı (%80 Teorik Hedef):       %{df_res['coverage_80_pct'].mean():.2f}")
    print(f"   • Ortalama Hata Payı (MAE):                            %{df_res['mae_pct'].mean():.2f}")
    print("=" * 70)

if __name__ == "__main__":
    main()
