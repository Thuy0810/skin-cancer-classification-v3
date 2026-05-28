from __future__ import annotations

"""Airflow DAG for scheduled model retraining.

Place this file inside your Airflow DAGs folder after adjusting PROJECT_DIR.
"""

from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT_DIR = "/opt/airflow/skin-cancer-classification"
CONFIG_PATH = "configs/train_config.yaml"

with DAG(
    dag_id="ham10000_retrain_pipeline",
    description="Validate data, prepare parquet, train, evaluate and serve HAM10000 model.",
    start_date=datetime(2026, 1, 1),
    schedule="@weekly",
    catchup=False,
    tags=["ml", "medical-image", "ham10000"],
) as dag:
    validate_data = BashOperator(
        task_id="validate_data",
        bash_command=f"cd {PROJECT_DIR} && python -m skin_cancer.data.validation --config {CONFIG_PATH}",
    )

    prepare_data = BashOperator(
        task_id="prepare_data",
        bash_command=f"cd {PROJECT_DIR} && python -m skin_cancer.data.preparation --config {CONFIG_PATH}",
    )

    train_model = BashOperator(
        task_id="train_model",
        bash_command=f"cd {PROJECT_DIR} && python -m skin_cancer.training.train --config {CONFIG_PATH} --use-weighted-sampler",
    )

    evaluate_model = BashOperator(
        task_id="evaluate_model",
        bash_command=f"cd {PROJECT_DIR} && python -m skin_cancer.evaluation.evaluate --config {CONFIG_PATH}",
    )

    validate_data >> prepare_data >> train_model >> evaluate_model
