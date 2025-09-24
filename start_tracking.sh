export MLFLOW_S3_ENDPOINT_URL=https://storage.yandexcloud.net
export AWS_ACCESS_KEY_ID=$S3_ACCESS_KEY
export AWS_SECRET_ACCESS_KEY=$S3_SECRET_KEY

mlflow server \
  --backend-store-uri postgresql://mle_20250822_3251e459d3_freetrack:33fd5e17432348ea9248bb6edc30cb89@rc1b-uh7kdmcx67eomesf.mdb.yandexcloud.net:6432/playground_mle_20250822_3251e459d3 \
  --registry-store-uri postgresql://mle_20250822_3251e459d3_freetrack:33fd5e17432348ea9248bb6edc30cb89@rc1b-uh7kdmcx67eomesf.mdb.yandexcloud.net:6432/playground_mle_20250822_3251e459d3 \
  --default-artifact-root s3://s3-student-mle-20250822-3251e459d3-freetrack \
  --no-serve-artifacts