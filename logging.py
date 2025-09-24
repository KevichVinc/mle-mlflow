import psycopg2 as psycopg
import pandas as pd
import os
import mlflow

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

# определяем название таблицы, в которой хранятся наши данные
TABLE_NAME = "users_churn"


# эта конструкция создаёт контекстное управление для соединения с базой данных 
# оператор with гарантирует, что соединение будет корректно закрыто после выполнения всех операций с базой данных
# причём закрыто оно будет даже в случае ошибки при работе с базой данных
# это нужно, чтобы не допустить так называемую "утечку памяти"
with psycopg.connect(**connection) as conn:

# создаём объект курсора для выполнения запросов к базе данных 
# с помощью метода execute() выполняется SQL-запрос для выборки данных из таблицы TABLE_NAME
    with conn.cursor() as cur:
        cur.execute(f"SELECT * FROM {TABLE_NAME}")
				
				# извлекаем все строки, полученные в результате выполнения запроса
        data = cur.fetchall()

				# получаем список имён столбцов из объекта курсора
        columns = [col[0] for col in cur.description]

# создаём объект DataFrame из полученных данных и имён столбцов 
# это позволяет удобно работать с данными в Python с использованием библиотеки Pandas
df = pd.DataFrame(data, columns=columns)

columns = df.columns.tolist()
columns_str = ",".join(columns)

with open("columns_sol.txt", "w", encoding="utf-8") as f:
    f.write(columns_str)

counts_columns = [
    "type", "paperless_billing", "internet_service", "online_security", "online_backup", "device_protection",
    "tech_support", "streaming_tv", "streaming_movies", "gender", "senior_citizen", "partner", "dependents",
    "multiple_lines", "target"
]

stats = {}

for col in counts_columns:
    # посчитайте уникальные значения для колонок, где немного уникальных значений
    column_stat = df[col].value_counts()
    column_stat = {f"{col}_{key}": value for key, value in column_stat.items()}

    # обновите словарь stats
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

# задаём название эксперимента и имя запуска для логирования в MLflow

EXPERIMENT_NAME = "churn_fio"
RUN_NAME = "data_check"

# создаём новый эксперимент в MLflow с указанным названием 
# если эксперимент с таким именем уже существует, 
# MLflow возвращает идентификатор существующего эксперимента
experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)

with mlflow.start_run(run_name=RUN_NAME, experiment_id=experiment_id) as run:
    # получаем уникальный идентификатор запуска эксперимента
    run_id = run.info.run_id
    
    # логируем метрики эксперимента
    # предполагается, что переменная stats содержит словарь с метриками,
    # объявлять переменную stats не надо,
    # где ключи — это названия метрик, а значения — числовые значения метрик
    mlflow.log_metrics(stats) # ваш код здесь
    
    # логируем файлы как артефакты эксперимента — 'columns.txt' и 'users_churn.csv'
    mlflow.log_artifact("columns.txt", artifact_path="dataframe")
    mlflow.log_artifact("users_churn.csv", artifact_path="dataframe")

experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
# получаем данные о запуске эксперимента по его уникальному идентификатору
run = mlflow.get_run(run_id)


# проверяем, что статус запуска эксперимента изменён на 'FINISHED'
# это утверждение (assert) можно использовать для автоматической проверки того, 
# что эксперимент был завершён успешно
assert run.info.status == 'FINISHED'

# удаляем файлы 'columns.txt' и 'users_churn.csv' из файловой системы,
# чтобы очистить рабочую среду после логирования артефактов
os.remove('columns.txt')
os.remove('users_churn.csv')