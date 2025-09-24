import mlflow
import mlflow.sklearn

loaded_model = load_model(model_uri=model_info.model_uri)

# делаем предсказание на X_test (предполагается, что он у тебя уже есть)
model_predictions = loaded_model.predict(X_test)

# проверка типа
assert model_predictions.dtype == int

print(model_predictions[:10])