# Se importa MLflow yy uuid para el seguimiento y la gestión de modelos
import uuid

import mlflow

from homework.src._internals.calculate_metrics import calculate_metrics
from homework.src._internals.parse_argument import parse_argument
from homework.src._internals.prepare_data import prepare_data
from homework.src._internals.print_metrics import print_metrics
from homework.src._internals.save_model_if_better import save_model_if_better
from homework.src._internals.select_model import select_model

FILE_PATH = "data/winequality-red.csv"
TEST_SIZE = 0.25
RANDOM_STATE = 123456


def main():

    args = parse_argument()
    model = select_model(args)

    x_train, x_test, y_train, y_test = prepare_data(
        file_path=FILE_PATH,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    mlflow.set_tracking_uri("file:mlruns")
    mlflow.set_experiment("wine_quality_experiment")

    ## Se inica la ejecución de MLflow
    mlflow.set_experiment("wine_quality_experiment")
    run_name = f"{args.model}_{uuid.uuid4().hex[:8]}"
    with mlflow.start_run(run_name=run_name):

        # logs de parámetros
        mlflow.log_param("file_path", FILE_PATH)
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("model_type", args.model)

        # log de los parámetros específicos del modelo

        if args.model == "elasticnet":
            mlflow.log_param("alpha", args.alpha)
            mlflow.log_param("l1_ratio", args.l1_ratio)
        elif args.model == "knn":
            mlflow.log_param("n_neighbors", args.n_neighbors)

        model.fit(x_train, y_train)

        # log de las métricas de entrenamiento

        mse, mae, r2 = calculate_metrics(model, x_train, y_train)
        print_metrics("Training metrics", mse, mae, r2)

        mlflow.log_metric("train_mse", mse)
        mlflow.log_metric("train_mae", mae)
        mlflow.log_metric("train_r2", r2)

        # log de las metricas de test

        mse, mae, r2 = calculate_metrics(model, x_test, y_test)
        print_metrics("Testing metrics", mse, mae, r2)

        mlflow.log_metric("test_mse", mse)
        mlflow.log_metric("test_mae", mae)
        mlflow.log_metric("test_r2", r2)

        # Ya no es necesario guardar el modelo manualmente, MLflow lo hace automáticamente

        mlflow.sklearn.log_model(
            sk_model=model, artifact_path="model", input_example=x_train[:1]
        )


if __name__ == "__main__":
    main()
