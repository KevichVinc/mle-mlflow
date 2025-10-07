# autofeat_mlflow.py
# Скрипт: подготовка данных -> AutoFeatClassifier -> логирование в MLflow -> печать ключевых значений

import os
import pandas as pd
import numpy as np
import mlflow
from sklearn.model_selection import train_test_split
from autofeat import AutoFeatClassifier

# ====================== Настройки окружения (MLflow + объектное хранилище) ======================
# Пример для S3-совместимого хранилища (Яндекс Облако / MinIO / AWS S3 и т.п.)
os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL", "https://storage.yandexcloud.net")
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID", "")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY", "")
os.environ["S3_BUCKET_NAME"] = os.getenv("S3_BUCKET_NAME", "")
# Имя бакета (нужно только для вывода пользователю; MLflow сам возьмет его из URI трекинга)
bucket_name = os.getenv("S3_BUCKET_NAME", "")

# Локальный / удалённый MLflow Tracking Server
TRACKING_SERVER_HOST = os.getenv("MLFLOW_TRACKING_HOST", "127.0.0.1")
TRACKING_SERVER_PORT = int(os.getenv("MLFLOW_TRACKING_PORT", "5000"))
mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}")
mlflow.set_registry_uri(f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}")

# Используем уже существующий эксперимент
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "eda_churn")
RUN_NAME = os.getenv("MLFLOW_RUN_NAME", "autofeat_features")
artifact_path = os.getenv("MLFLOW_ARTIFACT_PATH", "afc")

# ====================== Загрузка / подготовка данных ======================
# Ожидается датафрейм df со столбцом begin_date, таргетом target и признаками из списка ниже.
# Вставьте свою загрузку данных вместо примера ниже.
# Пример-рыба: сгенерируем минимальный датафрейм, если переменная DATA_CSV задана — читаем из файла.
csv_path = os.getenv("DATA_CSV", "")
if csv_path and os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
else:
    # Минимальный мок датасета под структуру из задания (ЗАМЕНИТЕ это на вашу загрузку)
    rng = pd.date_range("2021-01-01", periods=500, freq="D")
    df = pd.DataFrame({
        "begin_date": rng,
        "paperless_billing": np.random.choice(["Yes", "No"], size=len(rng)),
        "payment_method": np.random.choice(["Mailed check", "Electronic check", "Bank transfer", "Credit card"], size=len(rng)),
        "internet_service": np.random.choice(["DSL", "Fiber optic", "No"], size=len(rng)),
        "online_security": np.random.choice(["Yes", "No", "No internet service"], size=len(rng)),
        "online_backup": np.random.choice(["Yes", "No", "No internet service"], size=len(rng)),
        "device_protection": np.random.choice(["Yes", "No", "No internet service"], size=len(rng)),
        "tech_support": np.random.choice(["Yes", "No", "No internet service"], size=len(rng)),
        "streaming_tv": np.random.choice(["Yes", "No", "No internet service"], size=len(rng)),
        "streaming_movies": np.random.choice(["Yes", "No", "No internet service"], size=len(rng)),
        "gender": np.random.choice(["Male", "Female"], size=len(rng)),
        "senior_citizen": np.random.choice([0, 1], size=len(rng)),
        "partner": np.random.choice(["Yes", "No"], size=len(rng)),
        "dependents": np.random.choice(["Yes", "No"], size=len(rng)),
        "multiple_lines": np.random.choice(["Yes", "No", "No phone service"], size=len(rng)),
        "monthly_charges": np.random.uniform(20, 120, size=len(rng)).round(2),
        "total_charges": np.random.uniform(20, 6000, size=len(rng)).round(2),
        "target": np.random.choice([0, 1], size=len(rng)),
    })

# ====================== Конфигурация признаков и таргета (из условия) ======================
cat_features = [
    'paperless_billing',
    'payment_method',
    'internet_service',
    'online_security',
    'online_backup',
    'device_protection',
    'tech_support',
    'streaming_tv',
    'streaming_movies',
    'gender',
    'senior_citizen',
    'partner',
    'dependents',
    'multiple_lines',
]
num_features = ["monthly_charges", "total_charges"]
features = cat_features + num_features

target_col = os.getenv("TARGET_COLUMN", "target")  # можно переопределить через ENV
if target_col not in df.columns:
    raise ValueError(f"Не найден столбец таргета '{target_col}' в df.columns")

# Приводим числовые колонки к числовому типу (на случай строк/пробелов)
df[num_features] = df[num_features].apply(pd.to_numeric, errors="coerce")

# ====================== Разбиение train/test по дате (как в задании) ======================
split_column = "begin_date"
test_size = float(os.getenv("TEST_SIZE", "0.2"))

if split_column not in df.columns:
    raise ValueError(f"Не найден столбец даты '{split_column}' в df.columns")

df = df.sort_values(by=[split_column]).reset_index(drop=True)

X = df[features]
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, shuffle=False
)

# ====================== AutoFeatClassifier: генерация признаков ======================
transformations = ('1/', 'log', 'abs', 'sqrt')
afc = AutoFeatClassifier(
    categorical_cols=cat_features,
    transformations=transformations,
    feateng_steps=1,
    n_jobs=-1,
)

# fit/transform
X_train_features = afc.fit_transform(X_train, y_train)
X_test_features = afc.transform(X_test)

# ====================== Логирование в MLflow ======================
exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if exp is None:
    raise RuntimeError(f"Эксперимент '{EXPERIMENT_NAME}' не найден. Создайте его заранее или проверьте имя.")

experiment_id = exp.experiment_id

with mlflow.start_run(run_name=RUN_NAME, experiment_id=experiment_id) as run:
    run_id = run.info.run_id

    # Логируем сам AutoFeatClassifier как артефакт (как в подсказке)
    mlflow.sklearn.log_model(afc, artifact_path=artifact_path)

    # Полезные параметры для воспроизводимости
    mlflow.log_params({
        "transformations": ",".join(transformations),
        "feateng_steps": 1,
        "n_jobs": -1,
        "split_column": split_column,
        "test_size": test_size,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features_in": X_train.shape[1],
        "n_features_train_out": X_train_features.shape[1],
        "n_features_test_out": X_test_features.shape[1],
        "experiment_name": EXPERIMENT_NAME,
        "run_name": RUN_NAME,
        "artifact_path": artifact_path,
    })

# ====================== Вывод требуемых значений ======================
print(f'bucket_name = "{os.getenv("S3_BUCKET_NAME", "")}"')
print(f"experiment_id = {experiment_id}")
print(f'run_id = "{run_id}"')
print(f'artifact_path = "{artifact_path}"')
print("[OK] AutoFeatClassifier обучен и залогирован в MLflow.")
