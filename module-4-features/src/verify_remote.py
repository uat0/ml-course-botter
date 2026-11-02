"""Проверка опубликованной модели на трёх контрольных сессиях."""
from io import BytesIO
from pathlib import Path
import hashlib
import json
import joblib
import numpy as np
import pandas as pd
import requests
from features import engineer_features

# Все артефакты берутся из одной версии общего репозитория курса.
BASE_URL = 'https://raw.githubusercontent.com/uat0/ml-course-botter/main/module-4-features'
def download(path):
    response = requests.get(f'{BASE_URL}/{path}', timeout=60)
    response.raise_for_status()
    return response.content

# Проверяем целостность перед десериализацией доверенного файла.
payload = download('models/best_model.pkl')
checksums = json.loads(download('models/checksums.json'))
assert hashlib.sha256(payload).hexdigest() == checksums['best_model.pkl']
model = joblib.load(BytesIO(payload))
columns = json.loads(download('models/feature_cols.json'))
assert columns == model.named_steps['preprocess'].get_feature_names_out().tolist()

# Сравниваем вероятности с эталоном, полученным до сериализации.
sessions = pd.read_csv(BytesIO(download('data/example_sessions.csv')))
expected = np.array(json.loads(download('models/example_predictions.json')))
actual = model.predict_proba(engineer_features(sessions))[:, 1]
np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
print(pd.DataFrame({'Expected': expected, 'Downloaded': actual, 'Prediction': (actual >= .5).astype(int)}))
print('OK: raw-загрузка, SHA-256, порядок признаков и три вероятности совпадают.')
