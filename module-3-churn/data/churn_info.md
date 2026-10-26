# 📄 Telco Customer Churn — описание датасета

**Источник:** https://www.kaggle.com/datasets/blastchar/telco-customer-churn
(оригинал — IBM Sample Data Sets, файл `WA_Fn-UseC_-Telco-Customer-Churn.csv`; здесь сохранён как `telco_churn.csv`)

**Задача:** бинарная классификация — уйдёт ли клиент телеком-оператора (`Churn`).

## Общие характеристики

| Параметр | Значение |
|----------|----------|
| Записей | 7043 |
| Столбцов | 21 (19 признаков + `customerID` + `Churn`) |
| Числовых признаков | 3 (`tenure`, `MonthlyCharges`, `TotalCharges`) |
| Категориальных признаков | 16 (включая `SeniorCitizen` 0/1) |
| Ушли (Yes) | 1869 (26.5%) |
| Остались (No) | 5174 (73.5%) |
| Пропуски | 11 пустых строк `" "` в `TotalCharges` (0.16%) — у клиентов с `tenure = 0` |

## Столбцы

| Столбец | Тип | Описание | Значения |
|---------|-----|----------|----------|
| customerID | str | Уникальный ID клиента | — |
| gender | str | Пол | Male / Female |
| SeniorCitizen | int | Пенсионер | 0 / 1 |
| Partner | str | Есть супруг(а) | Yes / No |
| Dependents | str | Есть иждивенцы | Yes / No |
| tenure | int | Месяцев с компанией | 0–72 |
| PhoneService | str | Телефонная связь | Yes / No |
| MultipleLines | str | Несколько линий | Yes / No / No phone service |
| InternetService | str | Тип интернета | DSL / Fiber optic / No |
| OnlineSecurity | str | Онлайн-защита | Yes / No / No internet service |
| OnlineBackup | str | Онлайн-бэкап | Yes / No / No internet service |
| DeviceProtection | str | Страховка устройства | Yes / No / No internet service |
| TechSupport | str | Техподдержка | Yes / No / No internet service |
| StreamingTV | str | Стриминг ТВ | Yes / No / No internet service |
| StreamingMovies | str | Стриминг фильмов | Yes / No / No internet service |
| Contract | str | Тип контракта | Month-to-month / One year / Two year |
| PaperlessBilling | str | Электронный счёт | Yes / No |
| PaymentMethod | str | Способ оплаты | Electronic check / Mailed check / Bank transfer (automatic) / Credit card (automatic) |
| MonthlyCharges | float | Ежемесячная плата, $ | 18.25–118.75 |
| TotalCharges | str → float | Общая сумма платежей, $ | 18.8–8684.8 (+ 11 пустых) |
| **Churn** | str | **Целевая:** ушёл в последний месяц | Yes / No |

## Предобработка (в notebook.ipynb, раздел 5)
- `TotalCharges` → `pd.to_numeric(errors='coerce')`, пропуски → медиана (1397.47)
- `Churn` → 1/0
- Новые признаки: `TenureYears`, `AvgMonthlyCharge`, `IsNewClient`, `HasMultipleServices`
- Категориальные → `LabelEncoder` (словарь кодировщиков — `models/label_encoders.pkl`)
- Разбиение 80/20, `stratify=y`, `random_state=42`

## Доля оттока по ключевым сегментам

| Сегмент | Доля ушедших |
|---------|--------------|
| Contract: Month-to-month / One year / Two year | 42.7% / 11.3% / 2.8% |
| tenure 0–6 мес / 4–6 лет | 52.9% / 9.5% |
| InternetService: Fiber optic / DSL / No | 41.9% / 19.0% / 7.4% |
| PaymentMethod: Electronic check / автоплатёж картой | 45.3% / 15.2% |
| OnlineSecurity: No / Yes | 41.8% / 14.6% |
