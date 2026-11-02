"""Полный расчёт: EDA, модели, CV, Optuna, SHAP, сохранение."""
import os
from pathlib import Path
os.chdir(Path(__file__).resolve().parents[1])

# Импорты и фиксированные настройки обеспечивают воспроизводимость запуска.
# JSON хранит метаданные, hashlib контролирует целостность файлов.
import json, hashlib, platform, time, warnings, sys
# Path обеспечивает переносимые относительные пути.
from pathlib import Path
import importlib.metadata as metadata
# NumPy используется для векторных формул и проверок равенства.
import numpy as np
# Pandas сохраняет имена и типы табличных признаков.
import pandas as pd
import matplotlib
# Статический backend позволяет запускать полный код без окна приложения.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
# Неиспользуемый виджет прогресса не влияет на расчёты.
warnings.filterwarnings('ignore', message='IProgress not found.*')
import joblib, optuna, shap, requests
# Таблицы и рисунки сохраняются в выводах ноутбука.
from IPython.display import display, Markdown
from IPython import get_ipython
if get_ipython() is not None:
    get_ipython().run_line_magic('matplotlib', 'inline')
# clone создаёт необученную модель с теми же параметрами.
from sklearn.base import clone
# ColumnTransformer разделяет обработку чисел и категорий.
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
# Imputer обучает статистики заполнения только на train.
from sklearn.impute import SimpleImputer
# Стратификация сохраняет долю покупок в разбиениях.
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, classification_report,
    RocCurveDisplay, ConfusionMatrixDisplay)
# Отдельные реализации бустинга сравниваются на одинаковых данных.
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Все пути считаются от папки модуля; обучение ограничено четырьмя потоками.
# Seed фиксирует случайность, THREADS ограничивает загрузку процессора.
SEED, THREADS, N_TRIALS = 42, 4, 30
ROOT = Path('.') if Path('data').exists() else Path('module-4-features')
for folder in ['models', 'examples']:
    (ROOT / folder).mkdir(parents=True, exist_ok=True)
np.random.seed(SEED)
# Единые подписи и стиль упрощают сопоставление графиков.
sns.set_theme(style='whitegrid', font='DejaVu Sans', palette='deep')
plt.rcParams.update({'figure.dpi': 110, 'savefig.dpi': 160, 'axes.spines.top': False,
                     'axes.spines.right': False})

# Сохраняем график и выводим его в ноутбуке, не оставляя открытые фигуры.
def finish_plot(name):
    # Изображение доступно отдельно от ноутбука для приложения к отчёту.
    plt.savefig(ROOT / 'examples' / name, bbox_inches='tight', facecolor='white')
    plt.show()
    plt.close('all')

# Загружаем именно сохранённую копию данных и проверяем её размер и контрольную сумму.
data_path = ROOT / 'data/online_shoppers_intention.csv'
df = pd.read_csv(data_path)
dataset_sha = hashlib.sha256(data_path.read_bytes()).hexdigest()
# Размер проверяется до анализа, чтобы не обучиться на другом файле.
assert df.shape == (12330, 18)
assert dataset_sha == 'b3055ee355f59134d851d32641183cb4a8b45def7124d2f50442a042f358e0d9'
print('Размер:', df.shape)
print(df.head().to_string(index=False))
print('SHA-256:', dataset_sha)
display(df.dtypes.rename('Тип').to_frame())

# Сравниваем число и долю классов в исходном файле.
# Положительный класс нельзя оценивать одной Accuracy.
class_counts = df['Revenue'].value_counts().sort_index()
display(pd.DataFrame({'Количество': class_counts, 'Доля': class_counts / len(df)}))

# Первичная диагностика не подменяется предположением об отсутствии ошибок.
df.info()
# Описательная статистика показывает диапазоны и редкие категории.
display(df.describe(include='all').T)
print('Пропусков:', int(df.isna().sum().sum()))
print('Полных дублей:', int(df.duplicated().sum()))
print('Значения Month:', sorted(df['Month'].unique()))

# Удаляем только точные дубли; проверяем отсутствие конфликтующих меток.
clean = df.drop_duplicates().copy()
# Целевая переменная исключается из проверки идентичности входов.
input_cols = df.columns.drop('Revenue').tolist()
# При конфликтующих метках обычного удаления дублей было бы недостаточно.
conflicts = clean.groupby(input_cols, dropna=False)['Revenue'].nunique().gt(1).sum()
assert conflicts == 0, 'Одинаковые входы с разными метками требуют группового разделения'
# Булева целевая переводится в числовые классы 0 и 1.
y = clean['Revenue'].astype(int)
X_raw = clean.drop(columns='Revenue')
# Индексы сохраняют связь с неизменённым исходным CSV.
train_idx, test_idx = train_test_split(clean.index, test_size=0.2, stratify=y,
                                      random_state=SEED)
# Пересечение train и test должно быть пустым.
assert set(train_idx).isdisjoint(test_idx)
eda = clean.loc[train_idx].copy()
# Проверяем размер и баланс классов после стратификации.
split_info = pd.DataFrame({'Размер': [len(train_idx), len(test_idx)],
    'Доля покупок': [y.loc[train_idx].mean(), y.loc[test_idx].mean()]}, index=['train', 'test'])
display(split_info)

# График пропусков использует исходный набор и показывает фактически нулевые значения.
plt.figure(figsize=(12, 3))
df.isna().mean().mul(100).plot.bar(color='#386cb0')
plt.ylabel('Доля пропусков, %')
plt.title('Проверка пропусков в исходных столбцах')
finish_plot('missing_values.png')

# Шесть графиков раскрывают дисбаланс, распределения и конверсию на train.
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
# Частоты показывают масштаб дисбаланса покупателей и остальных сессий.
sns.countplot(data=eda, x='Revenue', ax=axes[0, 0], color='#386cb0')
axes[0, 0].set(title='Покупки и сессии без покупки', xlabel='Покупка', ylabel='Сессий')
# Логарифм делает видимым распределение с длинным правым хвостом.
eda['LogProductDuration'] = np.log1p(eda['ProductRelated_Duration'])
# Разделение по Revenue показывает различие распределений классов.
sns.histplot(data=eda, x='LogProductDuration', hue='Revenue', bins=40,
             common_norm=False, element='step', ax=axes[0, 1])
axes[0, 1].set(title='Время на товарных страницах', xlabel='log(1 + секунды)', ylabel='Сессий')
# Нули сохраняются, поскольку используется log1p, а не log.
eda['LogPageValues'] = np.log1p(eda['PageValues'])
sns.boxplot(data=eda, x='Revenue', y='LogPageValues', ax=axes[0, 2], color='#7fcdbb')
axes[0, 2].set(title='Ценность страниц по классам', xlabel='Покупка', ylabel='log(1 + PageValues)')

# Для категорий показываем наблюдаемую долю покупок без причинной интерпретации.
# Доля покупок нормирует сравнение групп разного размера.
visitor_rates = eda.groupby('VisitorType')['Revenue'].mean().mul(100).sort_values()
visitor_rates.plot.barh(ax=axes[1, 0], color='#7fcdbb')
axes[1, 0].set(title='Конверсия по типу посетителя', xlabel='Покупки, %', ylabel='')
# Месяцы располагаются хронологически, а не по алфавиту.
month_order = ['Feb', 'Mar', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
eda.groupby('Month')['Revenue'].mean().reindex(month_order).mul(100).plot.bar(
    ax=axes[1, 1], color='#386cb0', rot=35)
axes[1, 1].set(title='Конверсия по месяцам', xlabel='Месяц', ylabel='Покупки, %')
sns.histplot(data=eda, x='ExitRates', hue='Revenue', bins=35, element='step',
             common_norm=False, ax=axes[1, 2])
axes[1, 2].set(title='Показатель выходов', xlabel='ExitRates', ylabel='Сессий')
fig.tight_layout(pad=2)
finish_plot('eda_plots.png')

# Корреляции рассчитываются только для количественных признаков, а не кодов категорий.
numeric_original = ['Administrative', 'Administrative_Duration', 'Informational',
    'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration',
    'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay']
plt.figure(figsize=(11, 9))
# Спирмен оценивает монотонные связи без требования нормальности.
sns.heatmap(eda[numeric_original + ['Revenue']].corr(method='spearman'),
            cmap='vlag', center=0, vmin=-1, vmax=1, annot=True, fmt='.2f')
plt.title('Корреляции Спирмена: числовые признаки и покупка, train')
finish_plot('correlation_heatmap.png')

# Явная карта принимает также альтернативное английское сокращение Jun.
MONTH_MAP = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'June': 6,
             'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}

def engineer_features(frame):
    # Копия предотвращает изменение исходных данных при повторном вызове.
    out = frame.drop(columns=['Revenue'], errors='ignore').copy()
    # Булев выходной получает однозначное числовое представление.
    out['Weekend'] = out['Weekend'].astype(int)
    out['TotalPages'] = out[['Administrative', 'Informational', 'ProductRelated']].sum(axis=1)
    out['TotalDuration'] = out[['Administrative_Duration', 'Informational_Duration',
                               'ProductRelated_Duration']].sum(axis=1)
    # Сглаживание знаменателя исключает деление на ноль.
    # Единица в знаменателе задаёт сглаживание, а не настоящую среднюю цену.
    out['AvgPageValue'] = out['PageValues'] / (out['TotalPages'] + 1)
    out['BounceExitRatio'] = out['BounceRates'] / (out['ExitRates'] + 0.001)
    # Значение определяется только типом текущего посетителя.
    out['IsNewVisitor'] = out['VisitorType'].eq('New_Visitor').astype(int)
    out['IsWeekend'] = out['Weekend']
    # Неизвестный месяц станет NaN и будет обработан обученным imputer.
    out['MonthNum'] = out['Month'].map(MONTH_MAP)
    # Условная группа номинальных кодов не измеряет интенсивность трафика.
    out['IsHighTraffic'] = out['TrafficType'].gt(10).astype(int)
    out['DurationPerPage'] = out['TotalDuration'] / (out['TotalPages'] + 1)
    # Доля товара отделяет профиль интереса от длины сессии.
    out['ProductPageShare'] = out['ProductRelated'] / out['TotalPages'].clip(lower=1)
    # Доля времени дополняет простое число товарных страниц.
    out['ProductTimeShare'] = out['ProductRelated_Duration'] / out['TotalDuration'].clip(lower=1)
    # Временные и логарифмические преобразования используют только значения текущей строки.
    out['LogTotalDuration'] = np.log1p(out['TotalDuration'])
    out['MonthSin'] = np.sin(2 * np.pi * out['MonthNum'] / 12)
    out['MonthCos'] = np.cos(2 * np.pi * out['MonthNum'] / 12)
    return out

# Медианы, категории и параметры масштаба будут оцениваться внутри каждого train-фолда.
CATEGORICAL = ['Month', 'VisitorType', 'OperatingSystems', 'Browser', 'Region', 'TrafficType']
def make_preprocessor(frame):
    # Список категорий фиксируется предметным смыслом, а не dtype.
    cats = [c for c in CATEGORICAL if c in frame.columns]
    # Остальные столбцы обрабатываются как количественные признаки.
    nums = [c for c in frame.columns if c not in cats]
    num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')),
                         ('scaler', StandardScaler())])
    # Неизвестная категория не должна останавливать прогнозирование.
    cat_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                         ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    # Выход DataFrame сохраняет имена для интерпретации и проверки схемы.
    return ColumnTransformer([('numeric', num_pipe, nums), ('categorical', cat_pipe, cats)],
        verbose_feature_names_out=False).set_output(transform='pandas')

# Масштабирование для деревьев необязательно, но включено единообразно и сохраняется в scaler.pkl.
X = engineer_features(X_raw)
X_train, X_test = X.loc[train_idx], X.loc[test_idx]
y_train, y_test = y.loc[train_idx], y.loc[test_idx]
assert np.isfinite(X.select_dtypes('number').to_numpy()).all()
new_features = X.columns.difference(X_raw.columns).tolist()
print('Новых признаков:', len(new_features), new_features)
print('Train:', X_train.shape, 'Test:', X_test.shape)

# Базовые конфигурации фиксируются до просмотра результатов test.
# Вес положительного класса рассчитывается без доступа к test.
positive_weight = float((y_train == 0).sum() / (y_train == 1).sum())
estimators = {
    # Ограничение глубины и доли признаков сдерживает сложность RF.
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, max_features=0.3,
        class_weight='balanced', random_state=SEED, n_jobs=THREADS),
    # Гистограммный метод ускоряет сравнение при фиксированном seed.
    'XGBoost': XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.9, colsample_bytree=0.9, scale_pos_weight=positive_weight,
        random_state=SEED, n_jobs=THREADS, eval_metric='logloss', tree_method='hist'),
    # Регуляризация и глубина ограничивают листовой рост дерева.
    'LightGBM': LGBMClassifier(n_estimators=200, num_leaves=31, max_depth=7,
        learning_rate=0.05, class_weight='balanced', random_state=SEED,
        reg_alpha=0.001, reg_lambda=0.001, subsample_freq=1,
        n_jobs=THREADS, verbosity=-1),
    # Запись служебных файлов CatBoost отключена для чистоты каталога.
    'CatBoost': CatBoostClassifier(iterations=200, depth=6, learning_rate=0.1,
        auto_class_weights='Balanced', random_seed=SEED, thread_count=THREADS,
        verbose=False, allow_writing_files=False),
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
# Все семейства используют одни и те же пять пар train/validation.
cv_splits = list(cv.split(X_train, y_train))
fitted, cv_scores, fit_times, cv_times = {}, {}, {}, {}

# Pipeline повторно обучает всю предобработку на каждом фолде, предотвращая утечку.
for name, estimator in estimators.items():
    pipeline = Pipeline([('preprocess', make_preprocessor(X_train)), ('model', clone(estimator))])
    start = time.perf_counter()
    # ROC-AUC рассчитывается независимо на validation каждого фолда.
    cv_scores[name] = cross_val_score(pipeline, X_train, y_train, cv=cv_splits,
                                      scoring='roc_auc', n_jobs=1)
    cv_times[name] = time.perf_counter() - start
    # Финальная базовая модель обучается на всём train, но не на test.
    start = time.perf_counter()
    fitted[name] = pipeline.fit(X_train, y_train)
    # Измерение fit не включает время кросс-валидации.
    fit_times[name] = time.perf_counter() - start
    print(f'{name}: CV AUC={cv_scores[name].mean():.5f}, fit={fit_times[name]:.2f} с', flush=True)

# Семейство для Optuna выбирается исключительно по обучающей кросс-валидации.
selected_family = max(cv_scores, key=lambda name: cv_scores[name].mean())
# Разброс фолдов помогает оценить устойчивость среднего результата.
cv_table = pd.DataFrame([{'Модель': name, 'CV ROC-AUC': scores.mean(),
    'CV std': scores.std(ddof=1), 'Fit, с': fit_times[name], 'CV, с': cv_times[name]}
    for name, scores in cv_scores.items()]).sort_values('CV ROC-AUC', ascending=False)
display(cv_table.round(5))
print('Семейство для тюнинга:', selected_family)

# Все метрики вычисляются по фактическим предсказаниям на одном и том же test.
def evaluate(model, frame, target, name):
    # Непороговые оценки нужны для ROC-AUC и PR-AUC.
    probability = model.predict_proba(frame)[:, 1]
    # Порог фиксирован до сравнения моделей и не выбирается на test.
    prediction = (probability >= 0.5).astype(int)
    # Precision и Recall относятся к положительному классу покупки.
    return {'Model': name, 'Accuracy': accuracy_score(target, prediction),
        'Precision': precision_score(target, prediction, zero_division=0),
        'Recall': recall_score(target, prediction), 'F1': f1_score(target, prediction),
        'ROC-AUC': roc_auc_score(target, probability),
        'PR-AUC': average_precision_score(target, probability)}

# Единая функция предотвращает различие методик расчёта между моделями.
results = [evaluate(model, X_test, y_test, name) for name, model in fitted.items()]
results_df = pd.DataFrame(results)
results_df['Fit, sec'] = results_df['Model'].map(fit_times)
display(results_df.round(5))
print('Accuracy постоянного прогноза 0:', accuracy_score(y_test, np.zeros(len(y_test))))

# ROC-кривые сравниваются на одной шкале с линией случайного ранжирования.
fig, ax = plt.subplots(figsize=(8, 6))
for name, model in fitted.items():
    # Каждая кривая использует те же метки контрольной выборки.
    RocCurveDisplay.from_predictions(y_test, model.predict_proba(X_test)[:, 1], name=name, ax=ax)
# Диагональ обозначает случайное ранжирование.
ax.plot([0, 1], [0, 1], '--', color='gray')
ax.set(title='ROC-кривые четырёх базовых моделей: test', xlabel='False Positive Rate',
       ylabel='True Positive Rate')
finish_plot('roc_curves.png')

# Матрица показывается для модели, выбранной по CV, а не по test.
fig, ax = plt.subplots(figsize=(6, 5))
# Абсолютные количества ошибок дополняют усреднённые метрики.
ConfusionMatrixDisplay.from_predictions(y_test, fitted[selected_family].predict(X_test),
    display_labels=['Нет покупки', 'Покупка'], cmap='Blues', ax=ax, colorbar=False)
ax.set_title(f'{selected_family}: порог 0.5, test')
finish_plot('confusion_matrix_baseline.png')
print(classification_report(y_test, fitted[selected_family].predict(X_test), digits=4))

# Получаем имена после One-Hot и важность выбранной базовой модели.
base_pipeline = fitted[selected_family]
# После One-Hot число столбцов отличается от числа исходных входов.
encoded_cols = base_pipeline.named_steps['preprocess'].get_feature_names_out().tolist()
tree = base_pipeline.named_steps['model']
# Для LightGBM явно выбирается gain; для RF — impurity importance.
importance_values = (tree.booster_.feature_importance(importance_type='gain')
                     if selected_family == 'LightGBM' else tree.feature_importances_)
# Каждое значение важности связывается с конкретным столбцом.
importances = pd.DataFrame({'Feature': encoded_cols, 'Importance': importance_values})
importances = importances.sort_values('Importance', ascending=False)
display(importances.head(15))
plt.figure(figsize=(10, 7))
sns.barplot(data=importances.head(15), x='Importance', y='Feature', color='#386cb0')
plt.title(f'Топ-15 признаков: {selected_family}, встроенная важность')
finish_plot('feature_importance.png')

# Абляция использует только train и те же самые фолды, что и сравнение моделей.
# Исходное представление используется как контроль в абляции.
raw_train = X_raw.loc[train_idx].copy()
raw_train['Weekend'] = raw_train['Weekend'].astype(int)
# При удалении PageValues исключается также его производное.
ablation_frames = {'Исходные признаки': raw_train, 'С инженерией признаков': X_train,
    'Без PageValues и AvgPageValue': X_train.drop(columns=['PageValues', 'AvgPageValue'])}
ablation_scores = {}
for label, frame in ablation_frames.items():
    # Готовые CV-оценки полного набора повторно используются без лишнего обучения.
    if label == 'С инженерией признаков':
        ablation_scores[label] = cv_scores[selected_family]
    else:
        pipeline = Pipeline([('preprocess', make_preprocessor(frame)),
                              ('model', clone(estimators[selected_family]))])
        # Параметры модели и разбиения одинаковы для всех вариантов входа.
        ablation_scores[label] = cross_val_score(pipeline, frame, y_train, cv=cv_splits,
                                                 scoring='roc_auc', n_jobs=1)
ablation_table = pd.DataFrame([{'Набор': label, 'CV ROC-AUC': scores.mean(),
    'CV std': scores.std(ddof=1)} for label, scores in ablation_scores.items()])
display(ablation_table.round(5))
# Числовые результаты абляции доступны для независимого просмотра.
ablation_table.to_csv(ROOT / 'models/ablation.csv', index=False)

# Диапазоны ограничивают сложность и время поиска, не используя test.
# У каждого семейства своё пространство допустимых параметров.
def suggest_params(trial, family):
    if family == 'LightGBM':
        # Глубина определяет допустимое число листьев LightGBM.
        depth = trial.suggest_int('max_depth', 3, 9)
        # Число деревьев подбирается дискретно, чтобы ограничить время поиска.
        return {'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
            'max_depth': depth, 'num_leaves': trial.suggest_int('num_leaves', 7, min(63, 2**depth)),
            # Логарифмическая шкала исследует как малые, так и большие шаги бустинга.
            'learning_rate': trial.suggest_float('learning_rate', 0.015, 0.15, log=True),
            # Минимум объектов в листе ограничивает подгонку к редким сессиям.
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 80),
            # Подвыборка объектов может снижать корреляцию деревьев.
            'subsample': trial.suggest_float('subsample', 0.65, 1.0),
            # Подвыборка столбцов регулирует сложность бустинга.
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.65, 1.0),
            # L1-регуляризация исследуется в нескольких порядках величины.
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10, log=True),
            # L2-регуляризация штрафует слишком большие значения листьев.
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10, log=True)}
    if family == 'CatBoost':
        return {'iterations': trial.suggest_int('iterations', 100, 500, step=50),
            'depth': trial.suggest_int('depth', 3, 8),
            # Логарифмическая шкала исследует как малые, так и большие шаги бустинга.
            'learning_rate': trial.suggest_float('learning_rate', 0.015, 0.15, log=True),
            # Регуляризация CatBoost включена в сравнение параметров.
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 20, log=True),
            # Случайность выбора разбиений выступает дополнительной регуляризацией.
            'random_strength': trial.suggest_float('random_strength', 0, 2)}
    if family == 'XGBoost':
        # Число деревьев подбирается дискретно, чтобы ограничить время поиска.
        return {'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            # Логарифмическая шкала исследует как малые, так и большие шаги бустинга.
            'learning_rate': trial.suggest_float('learning_rate', 0.015, 0.15, log=True),
            # Подвыборка объектов может снижать корреляцию деревьев.
            'subsample': trial.suggest_float('subsample', 0.65, 1.0),
            # Подвыборка столбцов регулирует сложность бустинга.
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.65, 1.0),
            # Минимальный вес дочернего узла ограничивает сложность XGBoost.
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 15),
            # L2-регуляризация штрафует слишком большие значения листьев.
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 20, log=True)}
    # Число деревьев подбирается дискретно, чтобы ограничить время поиска.
    return {'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
        'max_depth': trial.suggest_int('max_depth', 5, 18),
        # Число объектов в листе сглаживает прогнозы Random Forest.
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 12),
        'max_features': trial.suggest_float('max_features', 0.3, 1.0)}

# Первая точка совпадает с baseline; остальные ищутся алгоритмом TPE.
baseline_trial_params = {
    'LightGBM': dict(n_estimators=200, max_depth=7, num_leaves=31, learning_rate=0.05,
        min_child_samples=20, subsample=1.0, colsample_bytree=1.0, reg_alpha=0.001, reg_lambda=0.001),
    'CatBoost': dict(iterations=200, depth=6, learning_rate=0.1, l2_leaf_reg=3.0, random_strength=1.0),
    'XGBoost': dict(n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.9,
        colsample_bytree=0.9, min_child_weight=1, reg_lambda=1.0),
    'Random Forest': dict(n_estimators=200, max_depth=10, min_samples_leaf=1, max_features=0.3)}

def objective(trial):
    # subsample_freq активирует подвыборки LightGBM; иначе subsample не используется.
    # Параметры trial передаются новому необученному экземпляру модели.
    params = suggest_params(trial, selected_family)
    model = clone(estimators[selected_family]).set_params(**params)
    if selected_family == 'LightGBM':
        model.set_params(subsample_freq=1)
    pipeline = Pipeline([('preprocess', make_preprocessor(X_train)), ('model', model)])
    scores = cross_val_score(pipeline, X_train, y_train, cv=cv_splits,
                             scoring='roc_auc', n_jobs=1)
    # Отдельные оценки фолдов сохраняются вместе с результатом испытания.
    trial.set_user_attr('fold_scores', scores.tolist())
    return float(scores.mean())

# Логи сохраняют ход оптимизации; независимых test-оценок внутри objective нет.
optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
# Исходная конфигурация включена в поиск как контрольная точка.
study.enqueue_trial(baseline_trial_params[selected_family])
start = time.perf_counter()
# Каждое из 30 испытаний оценивается на пяти train-фолдах.
study.optimize(objective, n_trials=N_TRIALS,
    callbacks=[lambda s, t: print(f'Trial {t.number+1:02d}/{N_TRIALS}: '
        f'CV={t.value:.5f}, best={s.best_value:.5f}', flush=True)])
tuning_seconds = time.perf_counter() - start

# Решение о сохранении baseline или tuned принимается до вычисления tuned test-метрик.
# Контрольная выборка не участвует в принятии решения о выборе tuned.
use_tuned = study.best_value > float(cv_scores[selected_family].mean())
best_params = study.best_params if use_tuned else {}
best_estimator = clone(estimators[selected_family]).set_params(**best_params)
if selected_family == 'LightGBM' and use_tuned:
    best_estimator.set_params(subsample_freq=1)
best_pipeline = Pipeline([('preprocess', make_preprocessor(X_train)), ('model', best_estimator)])
start = time.perf_counter()
# Итог обучается на всём train после завершения поиска.
best_pipeline.fit(X_train, y_train)
best_fit_seconds = time.perf_counter() - start
final_name = selected_family + (' (Optuna)' if use_tuned else ' (baseline)')
final_cv_auc = max(study.best_value, float(cv_scores[selected_family].mean()))
# Только после выбора параметров вычисляются итоговые test-метрики.
final_result = evaluate(best_pipeline, X_test, y_test, final_name)
display(pd.DataFrame([next(r for r in results if r['Model'] == selected_family), final_result]).round(5))
print('Лучшие параметры:', best_params)
print('Время поиска, с:', round(tuning_seconds, 2))
study.trials_dataframe().to_csv(ROOT / 'models/optuna_trials.csv', index=False)

# Сохраняем историю поиска и окончательную матрицу ошибок.
fig, ax = plt.subplots(figsize=(9, 4))
# История показывает каждое испытание и накопленный максимум.
trial_values = [t.value for t in study.trials]
# Линия лучшего результата не обязана расти на каждом испытании.
ax.plot(range(1, N_TRIALS+1), trial_values, 'o', alpha=0.6, label='Испытание')
# Линия лучшего результата не обязана расти на каждом испытании.
ax.plot(range(1, N_TRIALS+1), np.maximum.accumulate(trial_values), label='Лучший CV ROC-AUC')
ax.set(xlabel='Номер испытания', ylabel='Train-CV ROC-AUC', title='История Optuna: 30 испытаний')
ax.legend()
finish_plot('optuna_history.png')
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(y_test, best_pipeline.predict(X_test),
    display_labels=['Нет покупки', 'Покупка'], cmap='Blues', ax=ax, colorbar=False)
ax.set_title(final_name + ': test, порог 0.5')
finish_plot('confusion_matrix.png')

# Отделяем обученный классификатор от предобработки для TreeExplainer.
preprocessor = best_pipeline.named_steps['preprocess']
best_tree = best_pipeline.named_steps['model']
# Фиксированная подвыборка ограничивает стоимость объяснения.
shap_frame = X_test.sample(min(500, len(X_test)), random_state=SEED)
# SHAP получает ровно то представление, которое использует классификатор.
encoded = preprocessor.transform(shap_frame)
feature_cols = encoded.columns.tolist()
# Tree-path-dependent использует статистики обученного дерева.
explainer = shap.TreeExplainer(best_tree, feature_perturbation='tree_path_dependent')
explanation = explainer(encoded)

# Для RF некоторые версии SHAP возвращают отдельный выход для каждого класса.
if explanation.values.ndim == 3:
    explanation = explanation[:, :, 1]
shap_values = explanation.values
# Базовый выход приводится к одному значению для каждого объекта.
base_values = np.broadcast_to(np.asarray(explanation.base_values).squeeze(), (len(encoded),))

# Визуальное представление возвращает секунды и доли вместо стандартизованных значений.
# Отображаемые значения отделены от входов для расчёта SHAP.
display_frame = encoded.copy()
numeric_cols = preprocessor.transformers_[0][2]
# Масштабированные числа заменяются исходными единицами только в подписях.
display_frame.loc[:, numeric_cols] = shap_frame[numeric_cols].to_numpy()
shap.summary_plot(shap_values, display_frame, max_display=15, show=False, rng=np.random.default_rng(SEED))
plt.title('SHAP summary: окончательная модель')
finish_plot('shap_summary.png')
shap.summary_plot(shap_values, display_frame, plot_type='bar', max_display=15, show=False, rng=np.random.default_rng(SEED))
plt.title('Средний абсолютный вклад SHAP')
finish_plot('shap_bar.png')

# Сумма вкладов проверяется против выхода дерева, а единицы явно фиксируются.
# Прогноз Pipeline нужен для независимой проверки суммы вкладов.
probabilities = best_pipeline.predict_proba(shap_frame)[:, 1]
is_rf = selected_family == 'Random Forest'
explained_output = base_values + shap_values.sum(axis=1)
# Для RF объясняется вероятность, для бустинга — логарифм шансов.
expected_output = probabilities if is_rf else np.log(probabilities / (1 - probabilities))
assert np.allclose(explained_output, expected_output, atol=1e-4)
# Иллюстрация выбирается по наибольшей оценке в SHAP-подвыборке.
waterfall_idx = int(np.argmax(probabilities))
waterfall_exp = shap.Explanation(values=shap_values[waterfall_idx],
    base_values=base_values[waterfall_idx], data=display_frame.iloc[waterfall_idx].to_numpy(),
    feature_names=feature_cols)
shap.plots.waterfall(waterfall_exp, max_display=12, show=False)
finish_plot('shap_waterfall.png')

# Главный признак выбирается по среднему абсолютному SHAP, не по порядку столбцов.
# Средний модуль вклада ранжирует признаки по всему SHAP-сэмплу.
shap_importance = pd.Series(np.abs(shap_values).mean(axis=0), index=feature_cols).sort_values(ascending=False)
top_shap_feature = shap_importance.index[0]
shap.dependence_plot(top_shap_feature, shap_values, display_frame,
                     interaction_index=None, show=False)
plt.title(f'SHAP dependence: {top_shap_feature}')
finish_plot('shap_dependence.png')
display(shap_importance.head(10).rename('mean_abs_SHAP').to_frame())
print('Единицы SHAP:', 'вероятность' if is_rf else 'логарифм шансов (log-odds)')
print('Сессия:', int(shap_frame.index[waterfall_idx]), 'оценка покупки:', probabilities[waterfall_idx])
print('Проверка аддитивности SHAP: OK')

# Пять моделей сохраняются с компрессией; все они обучены только на train.
# Единые имена соответствуют структуре сдачи модуля.
artifact_names = {'Random Forest': 'rf_model.pkl', 'XGBoost': 'xgb_model.pkl',
                  'LightGBM': 'lgbm_model.pkl', 'CatBoost': 'cat_model.pkl'}
for name, model in fitted.items():
    joblib.dump(model, ROOT / 'models' / artifact_names[name], compress=3)
# Лучший Pipeline хранит и классификатор, и обученную предобработку.
joblib.dump(best_pipeline, ROOT / 'models/best_model.pkl', compress=3)
# Отдельный preprocessor полезен для проверки схемы входов.
joblib.dump(preprocessor, ROOT / 'models/preprocessor.pkl', compress=3)
joblib.dump(preprocessor.named_transformers_['numeric'].named_steps['scaler'],
            ROOT / 'models/scaler.pkl', compress=3)

# Записываем совместимые JSON: имена, версии, разделение и фактические метрики.
def save_json(name, value):
    (ROOT / 'models' / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
# Версии библиотек нужны для совместимой десериализации моделей.
versions = {p: metadata.version(p) for p in ['numpy','pandas','scikit-learn','xgboost',
    'lightgbm','catboost','optuna','shap','matplotlib','seaborn','joblib','requests']}
# Порядок закодированных признаков фиксируется явно.
save_json('feature_cols.json', feature_cols)
save_json('input_cols.json', X_train.columns.tolist())
# Индексы позволяют точно восстановить независимый test.
save_json('split_indices.json', {'train': train_idx.tolist(), 'test': test_idx.tolist()})
# CV, test и время обучения хранятся под разными ключами.
metrics = {'baseline_test': results, 'final_test': final_result,
    'baseline_cv': {n: {'mean': float(s.mean()), 'std': float(s.std(ddof=1)), 'folds': s.tolist()}
                    for n,s in cv_scores.items()},
    'selected_family': selected_family, 'best_model': final_name, 'final_cv_roc_auc': final_cv_auc,
    'best_params': best_params, 'optuna_best_cv_roc_auc': study.best_value, 'n_trials': N_TRIALS,
    'fit_seconds': fit_times, 'final_fit_seconds': best_fit_seconds,
    'tuning_seconds': tuning_seconds, 'threshold': 0.5}
save_json('metrics.json', metrics)
# Метаданные описывают данные, автора и правила эксперимента.
save_json('metadata.json', {'author': 'Боттер Вадим Валерьевич', 'group': 'ПКТб-23-1',
    'date': '2026-11-02', 'python': platform.python_version(), 'versions': versions,
    'dataset_sha256': dataset_sha, 'raw_rows': len(df), 'duplicates_removed': int(df.duplicated().sum()),
    'clean_rows': len(clean), 'train_rows': len(X_train), 'test_rows': len(X_test),
    'random_state': SEED, 'cv_folds': 5, 'new_features': new_features,
    'pipeline_input': 'engineer_features(raw_dataframe)', 'encoded_feature_count': len(feature_cols),
    'selection_rule': 'highest train-CV ROC-AUC; holdout is never used for selection'})

# Контрольный пример сырого входа позволяет проверить прогноз после загрузки в новой сессии.
# Контрольные сессии берутся из test и не используются для обучения.
example_raw = X_raw.loc[test_idx[:3]]
example_raw.to_csv(ROOT / 'data/example_sessions.csv', index=False)
# Эталонные вероятности рассчитываются до повторной загрузки.
example_expected = best_pipeline.predict_proba(engineer_features(example_raw))[:, 1]
save_json('example_predictions.json', example_expected.tolist())
# SHA-256 обнаруживает повреждение или подмену байтов артефакта.
manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (ROOT / 'models').glob('*.pkl')}
save_json('checksums.json', manifest)
print('Сохранены:', sorted(p.name for p in (ROOT / 'models').iterdir()))

# Локальная проверка сравнивает именно вероятности, а не только метки классов.
# Контрольный CSV заново проходит тот же расчёт признаков.
example_engineered = engineer_features(pd.read_csv(ROOT / 'data/example_sessions.csv'))
local_loaded = joblib.load(ROOT / 'models/best_model.pkl')
local_probabilities = local_loaded.predict_proba(example_engineered)[:, 1]
# Точное сравнение вероятностей строже сравнения меток.
np.testing.assert_allclose(local_probabilities, example_expected, rtol=0, atol=1e-12)
print('Локальная загрузка: вероятности трёх сессий совпадают.')

# Raw-ссылки указывают на общий репозиторий курса, а не на отдельный репозиторий модуля.
from io import BytesIO
BASE_URL = 'https://raw.githubusercontent.com/uat0/ml-course-botter/main/module-4-features'
def get_remote_file(relative_path):
    response = requests.get(f'{BASE_URL}/{relative_path}', timeout=60)
    # Ответ 404 или ошибка сервера не должны считаться загрузкой модели.
    response.raise_for_status()
    return response.content

# Проверка может запускаться отдельно после публикации, без повторного обучения.
def verify_remote_model():
    content = get_remote_file('models/best_model.pkl')
    # Удалённый манифест связывает опубликованную модель с её SHA-256.
    remote_checksums = json.loads(get_remote_file('models/checksums.json'))
    assert hashlib.sha256(content).hexdigest() == remote_checksums['best_model.pkl']
    # Десериализуется только модель из указанного доверенного репозитория.
    remote_model = joblib.load(BytesIO(content))
    remote_cols = json.loads(get_remote_file('models/feature_cols.json'))
    # Порядок One-Hot столбцов должен совпадать с сохранённой схемой.
    assert remote_cols == remote_model.named_steps['preprocess'].get_feature_names_out().tolist()
    remote_probabilities = remote_model.predict_proba(example_engineered)[:, 1]
    # Проверка выполняется на тех же трёх входах и допуске 1e-12.
    np.testing.assert_allclose(remote_probabilities, example_expected, rtol=0, atol=1e-12)
    display(pd.DataFrame({'Исходная вероятность': example_expected,
        'После raw-загрузки': remote_probabilities, 'Прогноз': (remote_probabilities >= .5).astype(int)}))
    print('Raw-загрузка: SHA-256, порядок признаков и вероятности совпадают.')

# Во время локальной подготовки проверяется существование опубликованного файла.
probe = requests.get(f'{BASE_URL}/models/best_model.pkl', timeout=60)
remote_verified = False
# Ожидание публикации явно отличается от успешной raw-проверки.
if probe.status_code == 404:
    print('Raw-проверка ожидает публикации файлов; локальная проверка пройдена.')
else:
    probe.raise_for_status()
    verify_remote_model()
    remote_verified = True
