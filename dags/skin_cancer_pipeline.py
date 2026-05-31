from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator


PROJECT_DIR = "/opt/airflow/project"

BASE_CMD = (
    f"cd {PROJECT_DIR} && "
    f"export PYTHONPATH={PROJECT_DIR}/src && "
)


EXPERIMENTS = [
    {
        "task_name": "b0_bs16_gamma1",
        "config": f"{PROJECT_DIR}/configs/eff_B0/b0_bs16_gamma1.yaml",
        "use_sampler": True,
    },
    # {
    #     "task_name": "b0_bs16_gamma2",
    #     "config": f"{PROJECT_DIR}/configs/eff_B0/b0_bs16_gamma2.yaml",
    #     "use_sampler": False,
    # },

    # {
    #     "task_name": "b0_bs32_gamma2",
    #     "config": f"{PROJECT_DIR}/configs/eff_B0/b0_bs32_gamma2.yaml",
    #     "use_sampler": False,
    # },

    # {
    #     "task_name": "b3_bs16_gamma2",
    #     "config": f"{PROJECT_DIR}/configs/eff_B3/b3_bs16_gamma2.yaml",
    #     "use_sampler": False,
    # },

    # {
    #     "task_name": "b3_bs8_gamma2",
    #     "config": f"{PROJECT_DIR}/configs/eff_B3/b3_bs8_gamma2.yaml",
    #     "use_sampler": False,
    # },
]


with DAG(
    dag_id="skin_cancer_training_pipeline",
    description="HAM10000 skin cancer classification multi-experiment pipeline",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    max_active_tasks=1,
    tags=["skin-cancer", "ham10000", "mlops"],
) as dag:

    start = EmptyOperator(task_id="start")

    validate_data = BashOperator(
        task_id="validate_data",
        bash_command=(
            BASE_CMD
            + f"python -m skin_cancer.data.validation "
            + f"--config {PROJECT_DIR}/configs/train_config.yaml"
        ),
        retries=1,
        retry_delay=timedelta(minutes=2),
    )

    prepare_data = BashOperator(
        task_id="prepare_data",
        bash_command=(
            BASE_CMD
            + f"python -m skin_cancer.data.preparation "
            + f"--config {PROJECT_DIR}/configs/train_config.yaml"
        ),
        retries=1,
        retry_delay=timedelta(minutes=2),
    )

    finish = EmptyOperator(task_id="finish")

    start >> validate_data >> prepare_data

    previous_task = prepare_data

    for exp in EXPERIMENTS:
        sampler_flag = " --use-weighted-sampler" if exp["use_sampler"] else ""

        train_task = BashOperator(
            task_id=f"train_{exp['task_name']}",
            bash_command=(
                BASE_CMD
                + f"python -m skin_cancer.training.train "
                + f"--config {exp['config']}"
                + sampler_flag
            ),
            retries=1,
            retry_delay=timedelta(minutes=2),
        )

        previous_task >> train_task
        previous_task = train_task

    previous_task >> finish