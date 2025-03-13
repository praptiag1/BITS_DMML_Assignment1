from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
from src.mlFlowProject.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.mlFlowProject.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from src.mlFlowProject.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline
from src.mlFlowProject.pipeline.stage_04_model_trainer import ModelTrainerTrainingPipeline
from src.mlFlowProject.pipeline.stage_05_model_evaluation import ModelEvaluationTrainingPipeline

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def run_data_ingestion():
    pipeline = DataIngestionTrainingPipeline()
    pipeline.main()

def run_data_validation():
    pipeline = DataValidationTrainingPipeline()
    pipeline.main()

def run_data_transformation():
    pipeline = DataTransformationTrainingPipeline()
    pipeline.main()

def run_model_trainer():
    pipeline = ModelTrainerTrainingPipeline()
    pipeline.main()

def run_model_evaluation():
    pipeline = ModelEvaluationTrainingPipeline()
    pipeline.main()

with DAG(
    'customer_churn_pipeline',
    default_args=default_args,
    description='Customer Churn Prediction Pipeline',
    schedule_interval=timedelta(days=1),
    start_date=datetime(2023, 1, 1),
    catchup=False,
) as dag:

    data_ingestion = PythonOperator(
        task_id='data_ingestion',
        python_callable=run_data_ingestion,
    )

    data_validation = PythonOperator(
        task_id='data_validation',
        python_callable=run_data_validation,
    )

    data_transformation = PythonOperator(
        task_id='data_transformation',
        python_callable=run_data_transformation,
    )

    model_trainer = PythonOperator(
        task_id='model_trainer',
        python_callable=run_model_trainer,
    )

    model_evaluation = PythonOperator(
        task_id='model_evaluation',
        python_callable=run_model_evaluation,
    )

    data_ingestion >> data_validation >> data_transformation >> model_trainer >> model_evaluation
