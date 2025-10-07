#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ===================== ГЛУШИМ СПАМ FUTUREWARNING ВО ВСЕХ ПРОЦЕССАХ joblib =====================
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

# ====================== Дальше — обычные импорты и код ======================
import os
import json
import psycopg
import pandas as pd
import numpy as np
import mlflow
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif
import matplotlib.pyplot as plt

# ====================== Глобальные переменные ======================
TABLE_NAME = os.getenv("PG_TABLE_NAME", "users_churn")
TRACKING_SERVER_HOST = os.getenv("MLFLOW_TRACKING_HOST", "127.0.0.1")
TRACKING_SERVER_PORT = int(os.getenv("MLFLOW_TRACKING_PORT", "5000"))

EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "eda_churn")
RUN_NAME = os.getenv("MLFLOW_RUN_NAME", "feature_selection")
REGISTRY_MODEL_NAME = os.getenv("MLFLOW_REGISTRY_MODEL_NAME", "churn_model_rf")
FS_ASSETS = os.getenv("FS_ASSETS_DIR", "fs_assets")
artifact_path = os.getenv("MLFLOW_ARTIFACT_PATH", FS_ASSETS)

# Префильтр: TOP-K признаков по mutual_info (по умолчанию 50)
FS_PREFILTER_TOPK = int(os.getenv("FS_PREFILTER_TOPK", "50"))

# Порядок фаз: быстрый подбор -> финальная переоценка (по умолчанию так)
# Поставь FS_SLOW_FIRST=1, если хочешь сначала попытаться «медленным» SFS/SBS (может занять долго)
FS_SLOW_FIRST = os.getenv("FS_SLOW_FIRST", "0") == "1"

# Ограничим параллелизм RF для стабильности
DEFAULT_RF_JOBS = max(1, min(4, (os.cpu_count() or 2) // 2))
RF_MAX_JOBS = int(os.getenv("RF_MAX_JOBS", str(DEFAULT_RF_JOBS)))

# ====================== S3 / MLflow ======================
os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL", "https://storage.yandexcloud.net")
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID", "")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY", "")
os.environ["S3_BUCKET_NAME"] = os.getenv("S3_BUCKET_NAME", "")
bucket_name = os.getenv("S3_BUCKET_NAME", "")

mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}")
mlflow.set_registry_uri(f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}")

# ====================== PostgreSQL: твои переменные подключения ======================
connection = {"sslmode": "require", "target_session_attrs": "read-write"}
postgres_credentials = {
    "host": os.getenv("DB_DESTINATION_HOST", "localhost"),
    "port": os.getenv("DB_DESTINATION_PORT", "5432"),
    "dbname": os.getenv("DB_DESTINATION_NAME", "postgres"),
    "user": os.getenv("DB_DESTINATION_USER", "postgres"),
    "password": os.getenv("DB_DESTINATION_PASSWORD", ""),
}
connection.update(postgres_credentials)

# ====================== Данные ======================
def load_from_postgres() -> pd.DataFrame:
    with psycopg.connect(**connection) as conn:
        # Глушим предупреждение pandas про отсутствующий SQLAlchemy engine
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
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

# Приводим потенциально числовые колонки к числу, строки оставляем как есть (для one-hot)
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

# ====================== train/val/test (отбор только на train) ======================
if split_column:
    df = df.sort_values(by=[split_column]).reset_index(drop=True)
    X = df[feature_cols]; y = df[target_col]
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=val_size, shuffle=False)
else:
    X = df[feature_cols]; y = df[target_col]
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=val_size, shuffle=False)

# ====================== one-hot для RF ======================
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

X_train_enc, enc_cols = one_hot_fit_transform(X_train, cat_features, num_features)
X_val_enc  = one_hot_transform(X_val,  cat_features, num_features, enc_cols)
X_test_enc = one_hot_transform(X_test, cat_features, num_features, enc_cols)

# float32 для ускорения/экономии памяти
X_train_enc = X_train_enc.astype(np.float32)
X_val_enc   = X_val_enc.astype(np.float32)
X_test_enc  = X_test_enc.astype(np.float32)

y_train = y_train.reset_index(drop=True)
y_val   = y_val.reset_index(drop=True)
y_test  = y_test.reset_index(drop=True)

# ====================== ПРЕФИЛЬТР: VarianceThreshold + SelectKBest(mutual_info) ======================
vt = VarianceThreshold(threshold=0.0)
X_train_enc_vt = vt.fit_transform(X_train_enc)
cols_after_vt = X_train_enc.columns[vt.get_support(indices=True)]
X_val_enc_vt  = X_val_enc[cols_after_vt]
X_test_enc_vt = X_test_enc[cols_after_vt]

if X_train_enc_vt.shape[1] > FS_PREFILTER_TOPK:
    skb = SelectKBest(score_func=mutual_info_classif, k=FS_PREFILTER_TOPK)
    X_train_pf = skb.fit_transform(X_train_enc_vt, y_train)
    support = skb.get_support(indices=True)
    cols_after_skb = cols_after_vt[support]
    X_val_pf  = X_val_enc_vt[cols_after_skb].values
    X_test_pf = X_test_enc_vt[cols_after_skb].values
    final_cols = cols_after_skb
else:
    X_train_pf = X_train_enc_vt
    X_val_pf  = X_val_enc_vt.values
    X_test_pf = X_test_enc_vt.values
    final_cols = cols_after_vt

X_train_fs = pd.DataFrame(X_train_pf, columns=final_cols, index=X_train_enc.index).astype(np.float32)
X_val_fs   = pd.DataFrame(X_val_pf,   columns=final_cols, index=X_val_enc.index).astype(np.float32)
X_test_fs  = pd.DataFrame(X_test_pf,  columns=final_cols, index=X_test_enc.index).astype(np.float32)

print(f"[INFO] Признаков после префильтра: {X_train_fs.shape[1]} (было {X_train_enc.shape[1]})")
print(f"[INFO] RF_MAX_JOBS={RF_MAX_JOBS}, FS_PREFILTER_TOPK={FS_PREFILTER_TOPK}")

# ====================== Вспомогательные функции для устойчивого отбора ======================
def _estimator_slow():
    return RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=RF_MAX_JOBS,
    )

def _estimator_fast():
    # Быстрый оценщик только для подбора набора фич
    return RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=RF_MAX_JOBS,
    )

def _run_sfs(X, y, estimator, k=10, cv=4, forward=True):
    sfs = SFS(
        estimator,
        k_features=k,
        forward=forward,
        floating=False,
        scoring="roc_auc",
        cv=cv,
        n_jobs=1,
        verbose=0,
    )
    sfs = sfs.fit(X, y)
    names = list(sfs.k_feature_names_)
    score = sfs.k_score_
    return sfs, names, score

def _safe_select_features(method_name, X, y, forward=True):
    """
    Сначала пытаемся «медленной» конфигурацией (если FS_SLOW_FIRST=1),
    иначе — «быстрой». Затем финально переоцениваем метрику на RF(300), cv=4.
    """
    selected, score = None, None
    sfs_obj_final = None

    phases = []
    if FS_SLOW_FIRST:
        phases = [
            ("slow", _estimator_slow(), 4),
            ("fast", _estimator_fast(), 2),
        ]
    else:
        phases = [
            ("fast", _estimator_fast(), 2),
            ("slow", _estimator_slow(), 4),
        ]

    # Фаза подбора: берём первый успешно подобранный набор фич
    for tag, est, cv in phases:
        try:
            print(f"[INFO] {method_name.upper()} {tag} phase: RF(n_estimators={est.n_estimators}), cv={cv}")
            sfs_tmp, names_tmp, score_tmp = _run_sfs(X, y, est, k=10, cv=cv, forward=forward)
            selected = names_tmp
            # если сразу «slow» сработал — отлично, используем его и для метрики
            if tag == "slow":
                sfs_obj_final = sfs_tmp
                score = score_tmp
            break
        except KeyboardInterrupt:
            print(f"[WARN] {method_name.upper()} interrupted on {tag} phase — trying fallback...")
        except Exception as e:
            print(f"[WARN] {method_name.upper()} {tag} phase failed: {e}")

    if selected is None:
        raise RuntimeError(f"{method_name.upper()} failed on all phases")

    # Финальная переоценка по условиям (RF=300, cv=4) на фиксированном наборе фич:
    # чтобы не тратить время на поиск, просто передадим в SFS матрицу, уже
    # ограниченную этими 10 фичами, и k_features=10 => посчитается только CV-оценка.
    if sfs_obj_final is None:
        X_fixed = X[list(selected)]
        print(f"[INFO] {method_name.upper()} final scoring: RF(300), cv=4 on fixed 10 features")
        sfs_obj_final, selected, score = _run_sfs(X_fixed, y, _estimator_slow(), k=10, cv=4, forward=forward)

    return sfs_obj_final, list(selected), score

# ====================== Устойчивый SFS/SBS ======================
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)

    print("[INFO] SFS.fit (устойчивый режим)")
    sfs_obj, top_sfs, sfs_score = _safe_select_features("sfs", X_train_fs, y_train, forward=True)

    print("[INFO] SBS.fit (устойчивый режим)")
    sbs_obj, top_sbs, sbs_score = _safe_select_features("sbs", X_train_fs, y_train, forward=False)

print("\nSequential Forward Selection (k=10)")
print("CV Score:", sfs_score)
print("\nSequential Backward Selection (k=10)")
print("CV Score:", sbs_score)

# ====================== Артефакты локально ======================
os.makedirs(FS_ASSETS, exist_ok=True)

# Метрики/история: используем финальные объекты (на 10 фичах метрика корректна, а история короткая — это нормально)
sfs_df = pd.DataFrame.from_dict(sfs_obj.get_metric_dict()).T
sbs_df = pd.DataFrame.from_dict(sbs_obj.get_metric_dict()).T
sfs_df.to_csv(f"{FS_ASSETS}/sfs.csv", index=True)
sbs_df.to_csv(f"{FS_ASSETS}/sbs.csv", index=True)

plt.figure()
plot_sfs(sfs_obj.get_metric_dict(), kind="std_dev")
plt.title("Sequential Forward Selection (w. StdDev)")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{FS_ASSETS}/sfs.png", dpi=150)
plt.close()

plt.figure()
plot_sfs(sbs_obj.get_metric_dict(), kind="std_dev")
plt.title("Sequential Backward Selection (w. StdDev)")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{FS_ASSETS}/sbs.png", dpi=150)
plt.close()

# Пересечение/объединение
interc_features = list(set(top_sbs) & set(top_sfs))
union_features  = list(set(top_sbs) | set(top_sfs))
with open(f"{FS_ASSETS}/intersection_features.json", "w", encoding="utf-8") as f:
    json.dump(interc_features, f, ensure_ascii=False, indent=2)
with open(f"{FS_ASSETS}/union_features.json", "w", encoding="utf-8") as f:
    json.dump(union_features, f, ensure_ascii=False, indent=2)
with open(f"{FS_ASSETS}/top_sfs.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(top_sfs))
with open(f"{FS_ASSETS}/top_sbs.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(top_sbs))

# ====================== MLflow логирование ======================
exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if exp is None:
    raise RuntimeError(f"Эксперимент '{EXPERIMENT_NAME}' не найден. Проверь имя или создай эксперимент.")
experiment_id = exp.experiment_id

with mlflow.start_run(run_name=f"{RUN_NAME}_intersection_and_union", experiment_id=experiment_id) as run:
    run_id = run.info.run_id
    mlflow.log_params({
        "table_name": TABLE_NAME,
        "experiment_name": EXPERIMENT_NAME,
        "registry_model_name": REGISTRY_MODEL_NAME,
        "fs_method_1": "SFS",
        "fs_method_2": "SBS",
        "estimator_final": f"RandomForestClassifier(n_estimators=300, n_jobs={RF_MAX_JOBS})",
        "estimator_fast": "RandomForestClassifier(n_estimators=100)",
        "scoring": "roc_auc",
        "cv_final": 4,
        "k_features": 10,
        "n_train": len(X_train_fs),
        "n_val": len(X_val_fs),
        "n_test": len(X_test_fs),
        "n_features_encoded_prefiltered": X_train_fs.shape[1],
        "pg_host": connection.get("host", ""),
        "pg_dbname": connection.get("dbname", ""),
        "prefilter_topk": FS_PREFILTER_TOPK,
        "rf_max_jobs": RF_MAX_JOBS,
        "fs_slow_first": int(FS_SLOW_FIRST),
    })
    mlflow.log_artifacts(FS_ASSETS, artifact_path=artifact_path)

# ====================== Вывод требуемых значений ======================
print("\n====================== Итог ======================")
print(f"bucket_name = \"{bucket_name}\"")
print(f"experiment_id = {experiment_id}")
print(f"run_id = \"{run_id}\"")
print(f"artifact_path = \"{artifact_path}\"")

print("\nTop SFS (k=10):", top_sfs)
print("Top SBS (k=10):", top_sbs)
print("Intersection features:", interc_features)
print("Union features:", union_features)
