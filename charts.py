import os
import psycopg
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow

os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://storage.yandexcloud.net" #endpoint бакета от YandexCloud
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID") # получаем id ключа бакета, к которому подключён MLFlow, из .env
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY") # получаем ключ бакета, к которому подключён MLFlow, из .env

TABLE_NAME = "users_churn"  # таблица с данными в Postgres

TRACKING_SERVER_HOST = "127.0.0.1"
TRACKING_SERVER_PORT = 5000

# ⚠️ ОБЯЗАТЕЛЬНО укажите непустое имя эксперимента!
# Можно прокинуть через переменную окружения MLFLOW_EXPERIMENT_NAME
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "eda_churn")  # замените на своё имя при необходимости
RUN_NAME = "eda"

ASSETS_DIR = "assets"
os.makedirs(ASSETS_DIR, exist_ok=True)

pd.options.display.max_columns = 100
pd.options.display.max_rows = 64

sns.set_style("white")
sns.set_theme(style="whitegrid")

connection = {"sslmode": "require", "target_session_attrs": "read-write"}
postgres_credentials = {
    "host": os.getenv("DB_DESTINATION_HOST"),
    "port": os.getenv("DB_DESTINATION_PORT"),
    "dbname": os.getenv("DB_DESTINATION_NAME"),
    "user": os.getenv("DB_DESTINATION_USER"),
    "password": os.getenv("DB_DESTINATION_PASSWORD"),
}
connection.update(postgres_credentials)

with psycopg.connect(**connection) as conn:
    with conn.cursor() as cur:
        cur.execute(f"SELECT * FROM {TABLE_NAME}")
        data = cur.fetchall()
        columns = [col[0] for col in cur.description]

df = pd.DataFrame(data, columns=columns)

# ====================== Категориальные графики (уникальные пользователи) ======================
fig, axs = plt.subplots(2, 2)
fig.set_size_inches(16.5, 12.5, forward=True)
fig.tight_layout(pad=1.6)

x = "type"
y = "customer_id"
stat = ["count"]
agg_df = df.groupby(x)[y].nunique().reset_index(name=stat[0])
sns.barplot(data=agg_df, x=x, y=stat[0], ax=axs[0, 0])
axs[0, 0].set_title(f'Count {y} by {x} in train dataframe')

x = "payment_method"
y = "customer_id"
agg_df = df.groupby(x)[y].nunique().reset_index(name=stat[0])
sns.barplot(data=agg_df, x=x, y=stat[0], ax=axs[1, 0])
axs[1, 0].set_title(f'Count {y} by {x} in train dataframe')
# корректный поворот подписей без предупреждений
axs[1, 0].tick_params(axis='x', labelrotation=45)

x = "internet_service"
y = "customer_id"
stat = ["count"]
agg_df = df.groupby(x)[y].nunique().reset_index(name=stat[0])
sns.barplot(data=agg_df, x=x, y=stat[0], ax=axs[0, 1])
axs[0, 1].set_title(f'Count {y} by {x} in train dataframe')

x = "gender"
y = "customer_id"
stat = ["count"]
agg_df = df.groupby(x)[y].nunique().reset_index(name=stat[0])
sns.barplot(data=agg_df, x=x, y=stat[0], ax=axs[1, 1])
axs[1, 1].set_title(f'Count {y} by {x} in train dataframe')

plt.savefig(os.path.join(ASSETS_DIR, 'cat_features_1'))

# ====================== Бинарные признаки: топ-сочетания и heatmap ======================
x = "customer_id"
binary_columns = [
    "online_security", 
    "online_backup", 
    "device_protection", 
    "tech_support",
    "streaming_tv",
    "streaming_movies",
    "senior_citizen",
    "partner",
    "dependents",
]
stat = ['count']

print(
    df.groupby(binary_columns).agg(stat[0])[x]
    .reset_index()
    .sort_values(by=x, ascending=False)
    .head(10)
)

heat_df = df[binary_columns].apply(pd.Series.value_counts).T
sns.heatmap(heat_df)
plt.savefig(os.path.join(ASSETS_DIR, 'cat_features_2_binary_heatmap'))

# ====================== Платежи по датам: среднее/медиана/мода ======================
x = "begin_date"
charges_columns = ["monthly_charges", "total_charges"]

# приведение begin_date к datetime и сортировка
if not pd.api.types.is_datetime64_any_dtype(df[x]):
    df[x] = pd.to_datetime(df[x], errors="coerce")
df = df.sort_values(x)

# работаем с копией без пропусков по платежам
df_charges = df.dropna(subset=charges_columns, how='any').copy()

stats = ["mean", "median", lambda s: s.mode().iloc[0] if not s.mode().empty else pd.NA]

charges_monthly_agg = (
    df_charges[[x, charges_columns[0]]]
    .groupby(x).agg(stats).reset_index()
)
charges_monthly_agg.columns = charges_monthly_agg.columns.droplevel()
charges_monthly_agg.columns = [x, "monthly_mean", "monthly_median", "monthly_mode"]

charges_total_agg = (
    df_charges[[x, charges_columns[1]]]
    .groupby(x).agg(stats).reset_index()
)
charges_total_agg.columns = charges_total_agg.columns.droplevel()
charges_total_agg.columns = [x, "total_mean", "total_median", "total_mode"]

fig, axs = plt.subplots(2, 1)
fig.tight_layout(pad=2.5)
fig.set_size_inches(6.5, 5.5, forward=True)

sns.lineplot(data=charges_monthly_agg, ax=axs[0], x=x, y='monthly_mean')
sns.lineplot(data=charges_monthly_agg, ax=axs[0], x=x, y='monthly_median')
sns.lineplot(data=charges_monthly_agg, ax=axs[0], x=x, y='monthly_mode')
axs[0].legend(["mean", "median", "mode"])
axs[0].set_title(f"Count statistics for {charges_columns[0]} by {x}")

sns.lineplot(data=charges_total_agg, ax=axs[1], x=x, y='total_mean')
sns.lineplot(data=charges_total_agg, ax=axs[1], x=x, y='total_median')
sns.lineplot(data=charges_total_agg, ax=axs[1], x=x, y='total_mode')
axs[1].legend(["mean", "median", "mode"])
axs[1].set_title(f"Count statistics for {charges_columns[1]} by {x}")

plt.savefig(os.path.join(ASSETS_DIR, 'charges_by_date'))

# ====================== Распределение целевой переменной ======================
ASSETS_DIR = os.path.join(os.getcwd(), "assets")  # оставим как в задании

x = "target"
target_agg = df[x].value_counts().reset_index()
target_agg.columns = [x, "count"]
sns.barplot(data=target_agg, x=x, y="count")
plt.title(f"{x} total distribution")
plt.savefig(os.path.join(ASSETS_DIR, 'target_count'))

# ====================== Зависимость target от даты и пола ======================
x = "begin_date"
target = "target"

fig, axs = plt.subplots(2, 2)
fig.tight_layout(pad=1.6)
fig.set_size_inches(16.5, 12.5, forward=True)

# 1) количество 1 по датам (если target бинарный 0/1)
target_agg_by_date = (
    df[[x, target]]
    .groupby(x)[target]
    .sum()
    .reset_index(name="target_count")
)
sns.lineplot(data=target_agg_by_date, x=x, y="target_count", ax=axs[0, 0])
axs[0, 0].set_title("Target count by begin date")

# 2) количество 0 и 1 по датам
target_agg_types = (
    df[[x, target, 'customer_id']]
    .groupby([x, target])['customer_id']
    .count()
    .reset_index(name='customer_count')
)
sns.lineplot(data=target_agg_types, x=x, y="customer_count", hue=target, ax=axs[0, 1])
axs[0, 1].set_title("Target count type by begin date")

# 3) конверсия по датам
conversion_agg = (
    df[[x, target]]
    .groupby(x)[target]
    .agg(['sum', 'count'])
    .reset_index()
)
conversion_agg['conv'] = (conversion_agg['sum'] / conversion_agg['count']).round(2)
sns.lineplot(data=conversion_agg, x=x, y="conv", ax=axs[1, 0])
axs[1, 0].set_title("Conversion value")

# 4) конверсия по датам + пол
conversion_agg_gender = (
    df[[x, target, 'gender']]
    .groupby([x, 'gender'])[target]
    .agg(['sum', 'count'])
    .reset_index()
)
conversion_agg_gender['conv'] = (conversion_agg_gender['sum'] / conversion_agg_gender['count']).round(2)
sns.lineplot(data=conversion_agg_gender, x=x, y='conv', hue='gender', ax=axs[1, 1])
axs[1, 1].set_title("Conversion value by gender")

plt.savefig(os.path.join(ASSETS_DIR, 'target_by_date'))

# ====================== Распределения платежей по целевой переменной ======================
charges = ["monthly_charges", "total_charges"]
target = "target"

fig, axs = plt.subplots(2, 1)
fig.tight_layout(pad=1.5)
fig.set_size_inches(6.5, 6.5, forward=True)

sns.histplot(data=df, x=charges[0], hue=target, kde=True, ax=axs[0])
axs[0].set_title(f"{charges[0]} distribution")

sns.histplot(data=df, x=charges[1], hue=target, kde=True, ax=axs[1])
axs[1].set_title(f"{charges[1]} distribution")

plt.savefig(os.path.join(ASSETS_DIR, 'chargest_by_target_dist'))

# ====================== Логирование артефактов в MLflow (с защитой) ======================
mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}")
mlflow.set_registry_uri(f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}")

try:
    if not EXPERIMENT_NAME or EXPERIMENT_NAME.strip() == "":
        raise ValueError("EXPERIMENT_NAME is empty. Set EXPERIMENT_NAME or MLFLOW_EXPERIMENT_NAME.")

    exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
    else:
        experiment_id = exp.experiment_id

    with mlflow.start_run(run_name=RUN_NAME, experiment_id=experiment_id) as run:
        mlflow.log_artifacts(ASSETS_DIR)
        run_id = run.info.run_id
        print(f"run_id = {run_id}")
    print(f"[MLflow] Artifacts logged from: {ASSETS_DIR}")

except Exception as e:
    # Если сервер MLflow не запущен/недоступен или имя пустое — не падаем, а даём пояснение
    print(f"[MLflow] Skipped logging artifacts: {e}")
