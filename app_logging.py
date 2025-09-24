import psycopg2 as psycopg
import pandas as pd
import os
import mlflow
from dotenv import load_dotenv

load_dotenv()

connection = {"sslmode": "require", "target_session_attrs": "read-write"}
postgres_credentials = {
    "host": "rc1b-uh7kdmcx67eomesf.mdb.yandexcloud.net", 
    "port": "6432",
    "dbname": "playground_mle_20250822_3251e459d3",
    "user": "mle_20250822_3251e459d3_freetrack",
    "password": "33fd5e17432348ea9248bb6edc30cb89",
}
assert all([var_value != "" for var_value in list(postgres_credentials.values())])

connection.update(postgres_credentials)

os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://storage.yandexcloud.net" #endpoint бакета от YandexCloud
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID") # получаем id ключа бакета, к которому подключён MLFlow, из .env
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY") # получаем ключ бакета, к которому подключён MLFlow, из .env

# определяем название таблицы, в которой хранятся наши данные
TABLE_NAME = "users_churn"

# определяем глобальные переменные
# поднимаем MLflow локально
TRACKING_SERVER_HOST = "127.0.0.1"
TRACKING_SERVER_PORT = 5000
mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}")

print("Tracking URI:", mlflow.get_tracking_uri())
exps = mlflow.search_experiments()
print([e.name for e in exps])  # должен быть 'churn_fio'

exp = mlflow.get_experiment_by_name("churn_fio")
print(exp)  # не None

runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
print(runs[["run_id","status","start_time","end_time"]].head())


# создаём подключение и выбираем данные
with psycopg.connect(**connection) as conn:
    with conn.cursor() as cur:
        cur.execute(f"SELECT * FROM {TABLE_NAME}")
        data = cur.fetchall()
        columns = [col[0] for col in cur.description]

# создаём DataFrame
df = pd.DataFrame(data, columns=columns)

# список колонок в строку
columns = df.columns.tolist()
columns_str = ",".join(columns)

# сохраняем список колонок в файл (теперь в columns.txt)
with open("columns.txt", "w", encoding="utf-8") as f:
    f.write(columns_str)

# сохраняем весь датасет для логирования
df.to_csv("users_churn.csv", index=False)

# собираем статистику
counts_columns = [
    "type", "paperless_billing", "internet_service", "online_security", "online_backup", "device_protection",
    "tech_support", "streaming_tv", "streaming_movies", "gender", "senior_citizen", "partner", "dependents",
    "multiple_lines", "target"
]

stats = {}

for col in counts_columns:
    column_stat = df[col].value_counts()
    column_stat = {f"{col}_{key}": value for key, value in column_stat.items()}
    stats.update(column_stat)

# числовые признаки
stats["data_length"] = df.shape[0]
stats["monthly_charges_min"] = df["monthly_charges"].min()
stats["monthly_charges_max"] = df["monthly_charges"].max()
stats["monthly_charges_mean"] = df["monthly_charges"].mean()
stats["monthly_charges_median"] = df["monthly_charges"].median()

stats["total_charges_min"] = df["total_charges"].min()
stats["total_charges_max"] = df["total_charges"].max()
stats["total_charges_mean"] = df["total_charges"].mean()
stats["total_charges_median"] = df["total_charges"].median()

stats["unique_customers_number"] = df["customer_id"].nunique()
stats["end_date_nan"] = df["end_date"].isna().sum()

# MLflow experiment/run
EXPERIMENT_NAME = "churn_fio"
RUN_NAME = "data_check"

# если эксперимент уже есть — используем его, если нет — создаём
exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if exp is None:
    experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
else:
    experiment_id = exp.experiment_id

with mlflow.start_run(run_name=RUN_NAME, experiment_id=experiment_id) as run:
    run_id = run.info.run_id

    # логируем метрики
    mlflow.log_metrics(stats)

    # логируем артефакты
    mlflow.log_artifact("columns.txt", artifact_path="dataframe")
    mlflow.log_artifact("users_churn.csv", artifact_path="dataframe")

# проверка статуса
run = mlflow.get_run(run_id)
assert run.info.status == "FINISHED"

# при желании можно удалить файлы после логирования
os.remove("columns.txt")
os.remove("users_churn.csv")
