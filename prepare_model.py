import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import boto3
import joblib  # важно: модель сохранялась через joblib.dump

# ==== 0) Конфиг из .env ====
load_dotenv()

S3_BUCKET_NAME   = os.environ["S3_BUCKET_NAME"]              # напр.: my-bucket
S3_MODEL_KEY     = os.environ["S3_MODEL_KEY"]                # напр.: models/fitted_flats_model.pkl
S3_TEST_KEY      = os.environ["S3_TEST_KEY"]                 # напр.: data/test.csv
TARGET_COL       = os.environ["TARGET_COL"]                  # напр.: target
S3_ENDPOINT_URL  = os.getenv("S3_ENDPOINT_URL", "https://storage.yandexcloud.net")
THRESHOLD        = float(os.getenv("PRED_THRESHOLD", "0.5")) # порог для бинаризации
PRED_TRANSFORM   = os.getenv("PRED_TRANSFORM", "clip_0_1")   # clip_0_1 | sigmoid | identity

# ==== 1) Клиент S3 и локальные пути ====
s3 = boto3.client("s3", endpoint_url=S3_ENDPOINT_URL)
MODEL_LOCAL_PATH = "/tmp/model.pkl"
TEST_LOCAL_PATH  = "/tmp/test.csv"

# ==== 2) Скачиваем из S3 ====
s3.download_file(S3_BUCKET_NAME, S3_MODEL_KEY, MODEL_LOCAL_PATH)
s3.download_file(S3_BUCKET_NAME, S3_TEST_KEY, TEST_LOCAL_PATH)

# ==== 3) Загружаем pipeline через joblib ====
model = joblib.load(MODEL_LOCAL_PATH)  # это sklearn.Pipeline с шагами preprocessor -> CatBoostRegressor
if not hasattr(model, "predict"):
    raise TypeError("Загруженный объект не имеет метода .predict — проверь S3_MODEL_KEY, это точно joblib-модель?")

# ==== 4) Читаем тестовый CSV (один файл с таргетом) ====
df_test = pd.read_csv(TEST_LOCAL_PATH)
if TARGET_COL not in df_test.columns:
    raise ValueError(f"TARGET_COL={TARGET_COL} не найден в {TEST_LOCAL_PATH}")

y_test = df_test[TARGET_COL].to_numpy()
X_test = df_test.drop(columns=[TARGET_COL])

# ==== 5) Предсказания регрессора -> proba/prediction ====
raw = np.asarray(model.predict(X_test), dtype=float).ravel()

def to_proba(arr: np.ndarray, mode: str) -> np.ndarray:
    if mode == "sigmoid":
        return 1.0 / (1.0 + np.exp(-arr))
    if mode == "identity":
        return arr
    # по умолчанию — клип в [0,1], если регрессор обучался по вероятностям
    return np.clip(arr, 0.0, 1.0)

proba = to_proba(raw, PRED_TRANSFORM)            # shape: (n_samples,)
prediction = (proba >= THRESHOLD).astype(int)    # бинаризация 0/1

# Теперь готовы нужные переменные по ТЗ:
# model, X_test, y_test, proba, prediction
print(X_test.shape, y_test.shape, proba.shape, prediction.shape)

