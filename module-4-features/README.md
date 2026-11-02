# Модуль 4 — Инженерия признаков

**Выполнил:** Боттер Вадим Валерьевич, ПКТб-23-1  
**Преподаватель:** Спирин Игорь Сергеевич  
**Дата:** 02.11.2026  
**Данные:** [Online Shoppers Purchasing Intention, UCI](https://doi.org/10.24432/C5F88Q), CC BY 4.0.

Полный отчёт из 14 разделов: [notebook.ipynb](notebook.ipynb).
Версия для просмотра в браузере: [report.html](report.html) (скачать и открыть).

## Результаты

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
| --- | --- | --- | --- | --- | --- | --- |
| Random Forest | 0.8853 | 0.6054 | 0.7670 | 0.6767 | 0.9308 | 0.7163 |
| XGBoost | 0.8869 | 0.6167 | 0.7330 | 0.6699 | 0.9269 | 0.7289 |
| LightGBM | 0.8701 | 0.5599 | 0.7958 | 0.6573 | 0.9305 | 0.7400 |
| CatBoost | 0.8710 | 0.5612 | 0.8037 | 0.6609 | 0.9325 | 0.7398 |
| Random Forest (Optuna) | 0.8771 | 0.5745 | 0.8272 | 0.6781 | 0.9349 | 0.7489 |

Итог — **Random Forest (Optuna)**. Семейство и параметры выбраны по 5-fold train-CV,
ROC-AUC=0.93473. Test не используется для выбора. Порог фиксирован: 0.5.
После удаления 125 дублей: 12205 строк; train=9764,
test=2441. Полное сравнение, времена и параметры — в [metrics.json](models/metrics.json).

## Инженерия признаков

14 производных: суммы страниц и времени, отношения, доли, новый посетитель, выходной,
месяц и его sin/cos, log-время, условная группа TrafficType > 10.
Наблюдаемый CV-эффект новых признаков: +0.00136. Без PageValues и AvgPageValue
CV ROC-AUC=0.77456.
TrafficType — код категории, поэтому IsHighTraffic не означает реальный объём трафика.

PageValues и агрегаты завершённой сессии требуют аудита доступности до прогноза.
Веса классов нарушают калибровку; predict_proba используется как оценка для ранжирования.

## Запуск

Проверено на Python 3.12.14. Из папки модуля:

```bash
python -m pip install -r requirements.txt
python src/run_analysis.py
```

Либо открыть `notebook.ipynb` в Jupyter/VS Code и выполнить все ячейки.
Точное окружение: `requirements-lock.txt`. Полный запуск включает 30 испытаний по 5 фолдов.

## Предсказание сохранённой моделью

```python
import joblib
import pandas as pd
from src.features import engineer_features

model = joblib.load('models/best_model.pkl')
raw = pd.read_csv('data/example_sessions.csv')
probability = model.predict_proba(engineer_features(raw))[:, 1]
print(probability)
```

Модель содержит всю обучаемую предобработку. Повторно применять scaler не требуется.
Загружать pickle следует только из доверенного источника.

## Проверка из репозитория

```bash
python src/verify_remote.py
```

Скрипт скачивает модель, список признаков и три примера по raw-ссылкам `main`, проверяет
HTTP-статусы, SHA-256 и равенство вероятностей при допуске 1e-12.

## Файлы

- `notebook.ipynb`, `report.html` — выполненный отчёт.
- `data/` — исходный CSV, описание и три тестовых примера.
- `models/` — четыре базовых Pipeline, лучший Pipeline, scaler и preprocessor,
  JSON с метриками, версиями, разбиением и контрольными суммами, история Optuna и абляция.
- `examples/` — EDA, таблица метрик, ROC, матрицы ошибок, importance, Optuna и четыре SHAP-графика.
- `src/` — полный код, расчёт признаков и независимая raw-проверка.

Источники и ограничения приведены в разделе 14 ноутбука; описание лицензии — в `data/dataset_info.md`.
