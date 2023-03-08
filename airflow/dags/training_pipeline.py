from asyncio import tasks
import json
from airflow import DAG
from airflow.operators.python import PythonOperator
import pendulum
import os
from textwrap import dedent

with DAG("sensor_training", start_date=pendulum.datetime(2023, 3, 6, tz="UTC"),
    schedule_interval="@weekly", catchup=False,description='Sensor fault detection',default_args={'retries':2},tags=['example']) as dag:


    def training(**kwargs):
        from sensor.pipeline.training_pipeline import start_training_pipeline
        start_training_pipeline()

    def sync_artifact_to_s3_bucket(**kwargs):
        bucket_name=os.getenv('BUCKET_NAME')
        os.system(f" aws s3 sync /app/artifact s3://{bucket_name}/artifcats")
        os.system(f"aws s3 sync /app/saved_models  s3://{bucket_name}/saved_models")
    
    trainining_pipeline=PythonOperator(task_id='train_pipeline',python_callable=training)
    sync_data_to_s3=PythonOperator(task_ide='sync_to_s3',python_callable=sync_artifact_to_s3_bucket)

    trainining_pipeline>>sync_artifact_to_s3