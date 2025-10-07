export MLFLOW_S3_ENDPOINT_URL="https://storage.yandexcloud.net"
# если трекинг-сервер/MLflow указывает на S3 по HTTPS — это всё

mlflow server \
  --backend-store-uri "postgresql://${DB_DESTINATION_USER}:${DB_DESTINATION_PASSWORD}@${DB_DESTINATION_HOST}:${DB_DESTINATION_PORT}/${DB_DESTINATION_NAME}" \
  --registry-store-uri "postgresql://${DB_DESTINATION_USER}:${DB_DESTINATION_PASSWORD}@${DB_DESTINATION_HOST}:${DB_DESTINATION_PORT}/${DB_DESTINATION_NAME}" \
  --default-artifact-root "s3://${S3_BUCKET_NAME}" \
  --serve-artifacts \
  --host 0.0.0.0 --port 5000

