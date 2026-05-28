from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator


PROJECT_DIR = "/opt/airflow/project"
CONFIG_PATH = f"{PROJECT_DIR}/configs/train_config.yaml"

BASE_CMD = (
    f"cd {PROJECT_DIR} && "
    f"export PYTHONPATH={PROJECT_DIR}/src && "
)


with DAG(
    dag_id="skin_cancer_training_pipeline",
    description="HAM10000 skin cancer classification training pipeline",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["skin-cancer", "ham10000", "mlops"],
) as dag:

    validate_data = BashOperator(
        task_id="validate_data",
        bash_command=(
            BASE_CMD
            + f"python -m skin_cancer.data.validation "
            + f"--config {CONFIG_PATH}"
        ),
    )

    prepare_data = BashOperator(
        task_id="prepare_data",
        bash_command=(
            BASE_CMD
            + f"python -m skin_cancer.data.preparation "
            + f"--config {CONFIG_PATH}"
        ),
    )

    train_model = BashOperator(
        task_id="train_model",
        bash_command=(
            BASE_CMD
            + f"python -m skin_cancer.training.train "
            + f"--config {CONFIG_PATH} "
            + f"--use-weighted-sampler"
        ),
    )

    evaluate_model = BashOperator(
        task_id="evaluate_model",
        bash_command=(
            BASE_CMD
            + f"python -m skin_cancer.evaluation.evaluate "
            + f"--config {CONFIG_PATH}"
        ),
    )

    validate_data >> prepare_data >> train_model >> evaluate_model