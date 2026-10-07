from airflow import DAG
from airflow.providers.common.sql.operators.sql import SQLExecuteQueryOperator as MySqlOperator
from airflow.operators.python import PythonOperator
from airflow.hooks.base import BaseHook

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import requests
import json
import csv
import os
import mysql.connector


# ============================================================
# DEFAULT ARGUMENTS
# ============================================================

default_args = {
    "owner": "airflow",
    "start_date": datetime(2026, 9, 1),
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}


# ============================================================
# 1. EXTRACT
# Open-Meteo Weather API
# ============================================================

def extract_api():

    latitude = -3.9985
    longitude = 122.5129

    url = "https://api.open-meteo.com/v1/forecast"

    sekarang = datetime.now(
        ZoneInfo("Asia/Makassar")
    )

    akhir = sekarang.replace(
        minute=0,
        second=0,
        microsecond=0
    )

    mulai = datetime(
        2026, 9, 1, 0, 0,
        tzinfo=ZoneInfo("Asia/Makassar")
    )

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "visibility",
        "timezone": "Asia/Makassar",
        "start_hour": mulai.strftime(
            "%Y-%m-%dT%H:%M"
        ),
        "end_hour": akhir.strftime(
            "%Y-%m-%dT%H:%M"
        )
    }

    response = requests.get(
        url,
        params=params,
        timeout=120
    )

    response.raise_for_status()

    data = response.json()

    hourly = data["hourly"]

    hasil_data = []

    for i in range(len(hourly["time"])):

        hasil_data.append({
            "datetime": hourly["time"][i],
            "visibility": hourly["visibility"][i]
        })

    hasil = {
        "source": "Open-Meteo Weather API",
        "location": {
            "name": "Kendari",
            "latitude": latitude,
            "longitude": longitude
        },
        "timezone": "Asia/Makassar",
        "total_records": len(hasil_data),
        "data": hasil_data
    }

    os.makedirs(
        "/usr/local/airflow/dags/data",
        exist_ok=True
    )

    with open(
        "/usr/local/airflow/dags/data/visibilitas_kendari.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            hasil,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("EXTRACT VISIBILITAS BERHASIL")


# ============================================================
# 2. TRANSFORM
# ============================================================

def transform_data():

    with open(
        "/usr/local/airflow/dags/data/visibilitas_kendari.json",
        "r",
        encoding="utf-8"
    ) as f:

        data_json = json.load(f)

    data = data_json["data"]

    csv_path = (
        "/usr/local/airflow/dags/data/"
        "transformed_visibilitas.csv"
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "datetime",
                "visibility"
            ]
        )

        writer.writeheader()
        writer.writerows(data)

    print("TRANSFORM VISIBILITAS BERHASIL")


# ============================================================
# 3. LOAD
# ============================================================

def load_to_mysql():

    mysql_conn = BaseHook.get_connection(
        "mysql-kualitas-udara"
    )

    conn = mysql.connector.connect(
        host=mysql_conn.host,
        port=mysql_conn.port or 3306,
        user=mysql_conn.login,
        password=mysql_conn.password,
        database=mysql_conn.schema
    )

    cursor = conn.cursor()

    query = """
    INSERT INTO visibilitas_kendari
    (
        datetime,
        visibility
    )
    VALUES (%s, %s)
    ON DUPLICATE KEY UPDATE
        visibility = VALUES(visibility)
    """

    with open(
        "/usr/local/airflow/dags/data/"
        "transformed_visibilitas.csv",
        "r",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            cursor.execute(
                query,
                (
                    row["datetime"],
                    float(row["visibility"])
                    if row["visibility"]
                    else None
                )
            )

    conn.commit()

    cursor.close()
    conn.close()

    print("LOAD VISIBILITAS BERHASIL")


# ============================================================
# DAG
# ============================================================

with DAG(
    dag_id="etl_visibilitas_kendari",

    default_args=default_args,

    schedule="0 * * * *",

    catchup=False,

    tags=[
        "ETL",
        "Visibilitas",
        "Open-Meteo",
        "Kendari"
    ],

) as dag:

    create_table = MySqlOperator(
        task_id="create_table_mysql",

        conn_id="mysql-kualitas-udara",

        sql="""
        CREATE TABLE IF NOT EXISTS visibilitas_kendari (

            id INT AUTO_INCREMENT PRIMARY KEY,

            datetime DATETIME NOT NULL UNIQUE,

            visibility FLOAT

        );
        """
    )

    extract = PythonOperator(
        task_id="extract_api",

        python_callable=extract_api
    )

    transform = PythonOperator(
        task_id="transform_data",

        python_callable=transform_data
    )

    load = PythonOperator(
        task_id="load_to_mysql",

        python_callable=load_to_mysql
    )

    create_table >> extract >> transform >> load