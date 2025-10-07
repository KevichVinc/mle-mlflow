#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ===== шумные FutureWarning от sklearn глушим заранее (унаследуется дочерним процессам) =====
import os as _os
_os.environ.setdefault(
    "PYTHONWARNINGS",
    "ignore::FutureWarning:sklearn.model_selection._validation"
)

import warnings
warnings.filterwarnings(
    "ignore",
    message="`fit_params` is deprecated and will be removed in version 1.6",
    category=FutureWarning,
    module="sklearn.model_selection._validation",
)

# ====================== Импорты ======================
import os
import json
import psycopg
import pandas as pd
import numpy as np
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ====================== Константы / окружение ======================
TABLE_NAME = os.getenv("PG_TABLE_NAME", "users_churn")

# Эксперименты под регистрацию
EXPERIMENT_NAME_INTERC = "feature_selection_intersection"
EXPERIMENT_NAME_UNION  = "feature_selection_union"

# Имя модели в реестре и ИМЯ ЗАПУСКА (валидатор ждёт именно его)
REGISTRY_MODEL_NAME = os.getenv("MLFLOW_REGISTRY_MODEL_NAME", "churn_rf_fs")
RUN_NAME_FOR_VALIDATOR = REGISTRY_MODEL_NAME  # <= важно!

# Где лежат артефакты предыдущего шага (intersection_features.json / union_features.json)
FS_ASSETS = os.getenv("FS_ASSETS_DIR", "fs_assets")

# MLflow трекинг
TRACKING_SERVER_HOST = os.getenv("MLFLOW_TRACKING_HOST", "127.0.0.1")
TRACKING_SERVER_PORT = int(os.getenv("MLFLOW_TRACKING_PORT", "5000"))
mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}")
mlflow.set_registry_uri(f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}")

# S3/объектное хранилище (если используется)
os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL", "https://storage.yandexcloud.net")
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID", "")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY", "")
os.environ["S3_BUCKET_NAME"] = os.getenv("S3_BUCKET_NAME", "")
bucket_name = os.getenv("S3_BUCKET_NAME", "")

# Ограничим параллелизм RF для стабильности
DEFAULT_RF_JOBS = max(1, min(4, (os.cpu_count() or 2) // 2))
RF_MAX_JOBS = int(os.getenv("RF_MAX_JOBS", str(DEFAULT_RF_JOBS)))

# ====================== Подключение к PostgreSQL (как в прошлом скрипте) ======================
connection = {"sslmode": "require", "target_session_attrs": "read-write"}
postgres_credentials = {
    "host": os.getenv("DB_DESTINATION_HOST", "localhost"),
    "port": os.getenv("DB_DESTINATION_PORT", "5432"),
    "dbname": os.getenv("DB_DESTINATION_NAME", "postgres"),
    "user": os.getenv("DB_DESTINATION_USER", "postgres"),
    "password": os.getenv("DB_DESTINATION_PASSWORD", ""),
}
connection.update(postgres_credentials)

# ====================== Утилиты ======================
def load_from_postgres() -> pd.DataFrame:
    with psycopg.connect(**connection) as conn:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)  # глушим предупреждение pandas про SQLAlchemy
            df = pd.read_sql(f'SELECT * FROM "{TABLE_NAME}"', conn)
    return df

def make_mock_df(n=500) -> pd.DataFrame:
    rng = pd.date_range("2021-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "begin_date": rng,
        "paperless_billing": np.random.choice(["Yes", "No"], size=n),
        "payment_method": np.random.choice(["Mailed check", "Electronic check", "Bank transfer", "Credit card"], size=n),
        "internet_service": np.random.choice(["DSL", "Fiber optic", "No"], size=n),
        "online_security": np.random.choice(["Yes", "No", "No internet service"], size=n),
        "online_backup": np.random.choice(["Yes", "No", "No internet service"], size=n),
        "device_protection": np.random.choice(["Yes", "No", "No internet service"], size=n),
        "tech_support": np.random.choice(["Yes", "No", "No internet service"], size=n),
        "streaming_tv": np.random.choice(["Yes", "No", "No internet service"], size=n),
        "streaming_movies": np.random.choice(["Yes", "No", "No internet service"], size=n),
        "gender": np.random.choice(["Male", "Female"], size=n),
        "senior_citizen": np.random.choice([0, 1], size=n),
        "partner": np.random.choice(["Yes", "No"], size=n),
        "dependents": np.random.choice(["Yes", "No"], size=n),
        "multiple_lines": np.random.choice(["Yes", "No", "No phone service"], size=n),
        "monthly_charges": np.random.uniform(20, 120, size=n).round(2),
        "total_charges": np.random.uniform(20, 6000, size=n).round(2),
        "target": np.random.choice([0, 1], size=n),
    })

def ensure_experiment(name: str) -> str:
    exp = mlflow.get_experiment_by_name(name)
    return exp.experiment_id if exp is not None else mlflow.create_experiment(name)

def one_hot_fit_transform(train_df: pd.DataFrame, cat_cols, num_cols):
    train_cat = pd.get_dummies(train_df[cat_cols], dummy_na=False)
    train_num = train_df[num_cols].apply(pd.to_numeric, errors="coerce")
    Xtr = pd.concat([train_num, train_cat], axis=1)
    return Xtr, list(Xtr.columns)

def one_hot_transform(df_any: pd.DataFrame, cat_cols, num_cols, ref_cols):
    cat = pd.get_dummies(df_any[cat_cols], dummy_na=False)
    num = df_any[num_cols].apply(pd.to_numeric, errors="coerce")
    Xany = pd.concat([num, cat], axis=1).reindex(columns=ref_cols, fill_value=0)
    return Xany

def evaluate_binary(y_true, y_proba, thr=0.5):
    from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    auc = roc_auc_score(y_true, y_proba)
    y_pred = (y_proba >= thr).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    return {"roc_auc": float(auc), "accuracy": float(acc), "precision": float(prec),
            "recall": float(rec), "f1": float(f1), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}

def train_and_register(
    X_train, y_train, X_val, y_val, X_test, y_test,
    feature_list, experiment_name, fs_set_label, registry_model_name
):
    # Подмножество колонок
    cols = [c for c in feature_list if c in X_train.columns]
    if not cols:
        raise RuntimeError("Пустой список признаков для обучения.")

    Xtr = X_train[cols].astype(np.float32)
    Xva = X_val[cols].astype(np.float32)
    Xte = X_test[cols].astype(np.float32)

    # Эксперимент
    exp_id = ensure_experiment(experiment_name)

    # Модель
    clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=RF_MAX_JOBS)
    clf.fit(Xtr, y_train)

    # Метрики
    m_train = evaluate_binary(y_train, clf.predict_proba(Xtr)[:,1])
    m_val   = evaluate_binary(y_val,   clf.predict_proba(Xva)[:,1])
    m_test  = evaluate_binary(y_test,  clf.predict_proba(Xte)[:,1])

    # Логирование + регистрация (ВАЖНО: run_name = registry name)
    with mlflow.start_run(run_name=RUN_NAME_FOR_VALIDATOR, experiment_id=exp_id) as run:
        run_id = run.info.run_id

        # Отметим тип набора признаков
        mlflow.set_tag("fs_set", fs_set_label)
        mlflow.log_param("fs_set", fs_set_label)

        mlflow.log_params({
            "n_estimators": 300,
            "random_state": 42,
            "rf_n_jobs": RF_MAX_JOBS,
            "n_features_used": len(cols),
        })
        # Метрики
        mlflow.log_metrics({f"train_{k}": v for k,v in m_train.items()})
        mlflow.log_metrics({f"val_{k}": v for k,v in m_val.items()})
        mlflow.log_metrics({f"test_{k}": v for k,v in m_test.items()})

        # Логируем и РЕГИСТРИРУЕМ модель
        mlflow.sklearn.log_model(clf, artifact_path="model")
        mv = mlflow.register_model(model_uri=f"runs:/{run_id}/model", name=registry_model_name)
        version_id = int(mv.version)

    return {
        "experiment_name": experiment_name,
        "run_name": RUN_NAME_FOR_VALIDATOR,
        "run_id": run_id,
        "registered_model_name": registry_model_name,
        "model_version_id": version_id,
        "metrics": {"train": m_train, "val": m_val, "test": m_test},
        "used_features": cols,
    }

# ====================== Загрузка данных ======================
try:
    df = load_from_postgres()
    print(f"[INFO] Данные загружены из PostgreSQL: {TABLE_NAME}, shape={df.shape}")
except Exception as e:
    warnings.warn(f"Не удалось загрузить из PostgreSQL ({e}). Использую мок-датасет.")
    df = make_mock_df()
    print(f"[INFO] Использован мок-датасет, shape={df.shape}")

target_col = os.getenv("TARGET_COLUMN", "target")
if target_col not in df.columns:
    raise ValueError(f"Не найден столбец таргета '{target_col}' в df.columns")

# Приводим потенциально числовые к числу (не ломаем строки — они для one-hot)
for col in df.columns:
    if df[col].dtype == "object":
        try:
            df[col] = pd.to_numeric(df[col])
        except Exception:
            pass

split_column = "begin_date" if "begin_date" in df.columns else None
test_size = float(os.getenv("TEST_SIZE", "0.2"))
val_size  = float(os.getenv("VAL_SIZE",  "0.2"))

drop_cols = [target_col] + ([split_column] if split_column else [])
feature_cols = [c for c in df.columns if c not in drop_cols]
cat_features = [c for c in feature_cols if df[c].dtype == "object"]
num_features = [c for c in feature_cols if c not in cat_features]

# Сплит
if split_column:
    df = df.sort_values(by=[split_column]).reset_index(drop=True)
    X = df[feature_cols]; y = df[target_col]
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=val_size, shuffle=False)
else:
    X = df[feature_cols]; y = df[target_col]
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=val_size, shuffle=False)

# One-hot (ВАЖНО: те же правила, что и в FS-скрипте)
X_train_enc, enc_cols = one_hot_fit_transform(X_train, cat_features, num_features)
X_val_enc  = one_hot_transform(X_val,  cat_features, num_features, enc_cols)
X_test_enc = one_hot_transform(X_test, cat_features, num_features, enc_cols)

# ====================== Читаем списки признаков из артефактов FS ======================
inter_path = os.path.join(FS_ASSETS, "intersection_features.json")
union_path = os.path.join(FS_ASSETS, "union_features.json")

if not (os.path.exists(inter_path) and os.path.exists(union_path)):
    raise FileNotFoundError(
        f"Не найдены артефакты FS: {inter_path} / {union_path}. "
        "Сначала запусти скрипт отбора признаков и убедись, что артефакты лежат в FS_ASSETS."
    )

with open(inter_path, "r", encoding="utf-8") as f:
    intersection_features = json.load(f)
with open(union_path, "r", encoding="utf-8") as f:
    union_features = json.load(f)

print(f"[INFO] Загружено признаков: intersection={len(intersection_features)}, union={len(union_features)}")

# ====================== Обучение и регистрация двух моделей ======================
res_interc = train_and_register(
    X_train_enc, y_train, X_val_enc, y_val, X_test_enc, y_test,
    feature_list=intersection_features,
    experiment_name=EXPERIMENT_NAME_INTERC,
    fs_set_label="intersection",
    registry_model_name=REGISTRY_MODEL_NAME,
)

res_union = train_and_register(
    X_train_enc, y_train, X_val_enc, y_val, X_test_enc, y_test,
    feature_list=union_features,
    experiment_name=EXPERIMENT_NAME_UNION,
    fs_set_label="union",
    registry_model_name=REGISTRY_MODEL_NAME,
)

# ====================== Печать требуемых переменных ======================
print("\n====================== Итог для формы ======================")
print(f'registred_model_name = "{REGISTRY_MODEL_NAME}"')
print(f'model_registred_name_interc = "{res_interc["registered_model_name"]}"')
print(f"model_version_id_interc = {res_interc['model_version_id']}")
print(f'run_name_interc = "{res_interc["run_name"]}"')
print(f'run_id_interc = "{res_interc["run_id"]}"')
print(f'model_registred_name_union = "{res_union["registered_model_name"]}"')
print(f"model_version_id_union = {res_union['model_version_id']}")
print(f'run_name_union = "{res_union["run_name"]}"')
print(f'run_id_union = "{res_union["run_id"]}"')

# Дополнительно: краткий сравнительный вывод ROC AUC
print("\n---- ROC AUC ----")
print("INTERSECTION:", res_interc["metrics"]["val"]["roc_auc"], "(val) |", res_interc["metrics"]["test"]["roc_auc"], "(test)")
print("UNION       :", res_union["metrics"]["val"]["roc_auc"], "(val) |", res_union["metrics"]["test"]["roc_auc"], "(test)")
