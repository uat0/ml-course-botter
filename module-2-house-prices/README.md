# 🏠 Module 2 — House Prices: предобработка и регрессия

**Автор:** Боттер В.В., ПКТб-23-1
**Дата обучения моделей:** 2026-10-12
**Датасет:** [House Prices — Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

## 📊 Результаты (отложенная выборка 20%, 292 домов)

| Модель | RMSE (log) | MAE (log) | R² | RMSE ($) | Время обучения, с |
|--------|-----------|-----------|-----|---------|------|
| Linear Regression | 0.1292 | 0.0880 | 0.9010 | $21,443 | 0.048 |
| Ridge | 0.1181 | 0.0844 | 0.9173 | $21,514 | 0.009 |
| Lasso | 0.1120 | 0.0801 | 0.9255 | $19,613 | 0.022 |
| **ElasticNet** | 0.1116 | 0.0800 | 0.9261 | $19,699 | 0.026 |
| XGBoost *(бонус)* | 0.1125 | 0.0788 | 0.9250 | $18,689 | 1.480 |
| LightGBM *(бонус)* | 0.1215 | 0.0807 | 0.9125 | $19,057 | 1.347 |
| Stacking *(бонус)* | 0.1077 | 0.0751 | 0.9312 | $18,223 | 6.323 |

**Лучшая обязательная модель:** ElasticNet (alpha=0.005, l1_ratio=0.7) → `models/best_model.pkl`
**Лучшая в целом:** Stacking (Ridge + Lasso + XGBoost).
Lasso отобрал 70 из 257 признаков.

## 🚀 Быстрый старт

```python
import joblib, json, requests
from io import BytesIO

BASE_URL = "https://raw.githubusercontent.com/GITHUB_USERNAME/ml-course-FAMILIA/main/module-2-house-prices"
model = joblib.load(BytesIO(requests.get(f"{BASE_URL}/models/best_model.pkl").content))
scaler = joblib.load(BytesIO(requests.get(f"{BASE_URL}/models/scaler.pkl").content))
feature_cols = requests.get(f"{BASE_URL}/models/feature_cols.json").json()
# предобработка «сырого» дома — функция preprocess() в notebook.ipynb (раздел 5),
# её параметры сохранены в models/preprocess_meta.json
```

## 🔧 Что сделано
- EDA: 8 графиков (распределение цены до/после log1p, пропуски, heatmap, boxplot, scatter, Q-Q)
- Пропуски: смысловые → "None"/0, LotFrontage → медиана по району, остальное → мода/медиана
- Выбросы: удалены 2 дома (GrLivArea > 4000 и цена < $300k)
- Целевая: log1p (skew 1.88 → 0.12)
- Feature Engineering: TotalSF, TotalBath, TotalPorch, HouseAge, RemodAge, Has*-флаги, OverallQual×TotalSF
- Кодирование: порядковые — ручной маппинг, номинальные — One-Hot (итого 257 признаков)
- Модели: LR, Ridge, Lasso, ElasticNet (GridSearchCV, 5-fold) + XGBoost, LightGBM, Stacking

## 📁 Структура
- `notebook.ipynb` — полный отчёт (14 разделов)
- `data/` — train.csv, test.csv, house_prices_info.md
- `models/` — модели (.pkl), scaler, feature_cols.json, metrics.json, preprocess_meta.json
- `examples/` — eda_plots.png, residuals.png, predictions_scatter.png
- `submission.csv` — предсказания для Kaggle (Stacking)
- `requirements.txt`
