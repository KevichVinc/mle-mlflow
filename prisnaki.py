# preprocessing_mlflow.py
# Скрипт: загрузка данных из Postgres -> препроцессинг (кат + числ) -> логирование препроцессора в MLflow

import os
import psycopg
import pandas as pd
import mlflow

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder,
    SplineTransformer,
    QuantileTransformer,
    RobustScaler,
    PolynomialFeatures,
    KBinsDiscretizer,
)

# ====================== Глобальные переменные ======================
TABLE_NAME = os.getenv("DB_TABLE_NAME", "users_churn")  # таблица с данными

TRACKING_SERVER_HOST = "127.0.0.1"
TRACKING_SERVER_PORT = 5000

# используем уже существующий эксперимент
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "eda_churn")
RUN_NAME = "preprocessing"
REGISTRY_MODEL_NAME = os.getenv("REGISTRY_MODEL_NAME", "users_churn_preprocessor")

# S3/объектное хранилище для артефактов MLflow (пример: Yandex Object Storage)
os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL", "https://storage.yandexcloud.net")
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID", "")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY", "")

# ====================== Загрузка данных из Postgres ======================
connection = {"sslmode": "require", "target_session_attrs": "read-write"}
postgres_credentials = {
    "host": os.getenv("DB_DESTINATION_HOST", "localhost"),
    "port": os.getenv("DB_DESTINATION_PORT", "5432"),
    "dbname": os.getenv("DB_DESTINATION_NAME", "postgres"),
    "user": os.getenv("DB_DESTINATION_USER", "postgres"),
    "password": os.getenv("DB_DESTINATION_PASSWORD", ""),
}
connection.update(postgres_credentials)

with psycopg.connect(**connection) as conn:
    with conn.cursor() as cur:
        cur.execute(f"SELECT * FROM {TABLE_NAME}")
        data = cur.fetchall()
        columns = [col[0] for col in cur.description]

df = pd.DataFrame(data, columns=columns)

# ====================== Просмотр типов и выделение категориальных колонок ======================
# Выбор нечисловых (строковых) колонок — правильный ответ из задания:
obj_df = df.select_dtypes(include="object")

# Категориальные признаки из задания
cat_columns = ["type", "payment_method", "internet_service", "gender"]

# ====================== Задание 1: OneHotEncoder для категориальных ======================
encoder_oh = OneHotEncoder(
    categories="auto",
    handle_unknown="ignore",
    max_categories=10,
    sparse_output=False,
    drop="first",
)

# (опционально) Импутация категориальных, чтобы OHE не падал на пропусках
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", encoder_oh),
])

# ====================== Задание 2: Числовые признаки и энкодеры ======================
num_columns = ["monthly_charges", "total_charges"]

# на всякий случай приводим к числовому типу (строки/пробелы -> NaN)
df[num_columns] = df[num_columns].apply(pd.to_numeric, errors="coerce")

# гиперпараметры (можно переопределить через ENV)
n_knots = int(os.getenv("SPLINE_N_KNOTS", 5))
degree = int(os.getenv("POLY_DEGREE", 2))
n_quantiles = int(os.getenv("QUANT_N", 100))
n_bins = int(os.getenv("KBINS_N", 5))
strategy = os.getenv("KBINS_STRATEGY", "quantile")          # 'uniform' | 'quantile' | 'kmeans'
encode = os.getenv("KBINS_ENCODE", "onehot-dense")          # 'onehot' | 'onehot-dense' | 'ordinal'
subsample = int(os.getenv("KBINS_SUBSAMPLE", 200000))

# Каждый числовой трансформер получает сначала имputer, затем основной шаг
num_spl = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("spl", SplineTransformer(n_knots=n_knots)),
])

num_q = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("q", QuantileTransformer(n_quantiles=n_quantiles)),
])

num_rb = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("rb", RobustScaler()),
])

# без константного члена, чтобы не плодить "intercept" колонку
num_pol = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("pol", PolynomialFeatures(degree=degree, include_bias=False)),
])

num_kbd = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("kbd", KBinsDiscretizer(n_bins=n_bins, encode=encode, strategy=strategy, subsample=subsample)),
])

# ====================== Задание 3: Общий препроцессор (ColumnTransformer) ======================
# ВАЖНО: не используем вложенный ColumnTransformer для числовых,
# а кладем каждый пайплайн как отдельный "блок" по тем же num_columns.
# Так корректно формируются имена фич и не теряются колонки.
preprocessor = ColumnTransformer(
    transformers=[
        ('spl', num_spl, num_columns),
        ('q',   num_q,  num_columns),
        ('rb',  num_rb, num_columns),
        ('pol', num_pol, num_columns),
        ('kbd', num_kbd, num_columns),
        ('cat', categorical_transformer, cat_columns),
    ],
    remainder="drop",
    n_jobs=None,  # на время отладки лучше без параллелизма
)

# Обучаем и трансформируем
encoded_features = preprocessor.fit_transform(df)

# Получаем названия фич после преобразований
feature_names = preprocessor.get_feature_names_out()
transformed_df = pd.DataFrame(encoded_features, columns=feature_names, index=df.index)

# Объединяем с оригинальным df
df = pd.concat([df, transformed_df], axis=1)

# ====================== Логирование препроцессора в MLflow ======================
mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}")
mlflow.set_registry_uri(f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}")

exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if exp is None:
    raise RuntimeError(
        f"Эксперимент '{EXPERIMENT_NAME}' не найден. Создайте его заранее или проверьте имя."
    )

with mlflow.start_run(run_name=RUN_NAME, experiment_id=exp.experiment_id) as run:
    # Сохраняем общий препроцессор как артефакт (как в примере ТЗ — в директорию 'column_transformer')
    mlflow.sklearn.log_model(preprocessor, "column_transformer")
    mlflow.log_param("cat_columns", ",".join(cat_columns))
    mlflow.log_param("num_columns", ",".join(num_columns))
    mlflow.log_params(
        {
            "spline_n_knots": n_knots,
            "quantile_n": n_quantiles,
            "robust_scaler": True,
            "poly_degree": degree,
            "kbins_n_bins": n_bins,
            "kbins_strategy": strategy,
            "kbins_encode": encode,
        }
    )

print("[OK] Препроцессор обучен и залогирован в MLflow в папку 'column_transformer'.")

