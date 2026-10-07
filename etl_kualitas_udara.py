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
# Mengambil data langsung dari Open-Meteo API
# Mulai 1 September 2026 sampai waktu saat DAG dijalankan
# ============================================================

def extract_api():

    # --------------------------------------------------------
    # Lokasi Kendari
    # --------------------------------------------------------

    latitude = -3.9985
    longitude = 122.5129
    nama_lokasi = "Kendari, Sulawesi Tenggara"

    # --------------------------------------------------------
    # URL API
    # --------------------------------------------------------

    url = "https://air-quality-api.open-meteo.com/v1/air-quality"

    # --------------------------------------------------------
    # Waktu sekarang Asia/Makassar
    # --------------------------------------------------------

    waktu_sekarang = datetime.now(
        ZoneInfo("Asia/Makassar")
    )

    # Dibulatkan ke awal jam
    waktu_akhir = waktu_sekarang.replace(
        minute=0,
        second=0,
        microsecond=0
    )

    # Mulai dari 1 September 2026
    waktu_mulai = datetime(
        2026,
        9,
        1,
        0,
        0,
        tzinfo=ZoneInfo("Asia/Makassar")
    )

    # --------------------------------------------------------
    # Parameter API
    # --------------------------------------------------------

    params = {
        "latitude": latitude,
        "longitude": longitude,

        "hourly": (
            "pm10,"
            "pm2_5,"
            "carbon_monoxide,"
            "nitrogen_dioxide,"
            "sulphur_dioxide,"
            "ozone"
        ),

        "timezone": "Asia/Makassar",

        # Mulai 1 September
        "start_hour": waktu_mulai.strftime(
            "%Y-%m-%dT%H:%M"
        ),

        # Sampai jam saat DAG dijalankan
        "end_hour": waktu_akhir.strftime(
            "%Y-%m-%dT%H:%M"
        )
    }

    # --------------------------------------------------------
    # Request API
    # --------------------------------------------------------

    response = requests.get(
        url,
        params=params,
        timeout=120
    )

    response.raise_for_status()

    data_api = response.json()

    # --------------------------------------------------------
    # Mengambil bagian hourly
    # --------------------------------------------------------

    hourly = data_api.get("hourly", {})

    waktu = hourly.get("time", [])
    pm10 = hourly.get("pm10", [])
    pm2_5 = hourly.get("pm2_5", [])
    carbon_monoxide = hourly.get("carbon_monoxide", [])
    nitrogen_dioxide = hourly.get("nitrogen_dioxide", [])
    sulphur_dioxide = hourly.get("sulphur_dioxide", [])
    ozone = hourly.get("ozone", [])

    # --------------------------------------------------------
    # Membentuk data JSON
    # --------------------------------------------------------

    data_kualitas_udara = []

    for i in range(len(waktu)):

        data_kualitas_udara.append({
            "datetime": waktu[i],
            "pm10": pm10[i],
            "pm2_5": pm2_5[i],
            "carbon_monoxide": carbon_monoxide[i],
            "nitrogen_dioxide": nitrogen_dioxide[i],
            "sulphur_dioxide": sulphur_dioxide[i],
            "ozone": ozone[i]
        })

    # --------------------------------------------------------
    # Struktur JSON
    # --------------------------------------------------------

    hasil = {
        "source": "Open-Meteo Air Quality API",

        "location": {
            "name": nama_lokasi,
            "latitude": latitude,
            "longitude": longitude
        },

        "description": (
            "Hourly air quality data "
            "from 1 September 2026 "
            "until current time"
        ),

        "timezone": "Asia/Makassar",

        "start_time": waktu_mulai.strftime(
            "%Y-%m-%d %H:%M:%S"
        ),

        "end_time": waktu_akhir.strftime(
            "%Y-%m-%d %H:%M:%S"
        ),

        "total_records": len(data_kualitas_udara),

        "data": data_kualitas_udara
    }

    # --------------------------------------------------------
    # Lokasi JSON
    # --------------------------------------------------------

    output_path = (
        "/usr/local/airflow/dags/data/"
        "kualitas_udara.json"
    )

    os.makedirs(
        "/usr/local/airflow/dags/data",
        exist_ok=True
    )

    # --------------------------------------------------------
    # Simpan JSON
    # --------------------------------------------------------

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            hasil,
            file,
            ensure_ascii=False,
            indent=2
        )

    # --------------------------------------------------------
    # Log
    # --------------------------------------------------------

    print("=" * 60)
    print("EXTRACT API BERHASIL")
    print("=" * 60)

    print(
        "Sumber       : Open-Meteo Air Quality API"
    )

    print(
        f"Lokasi       : {nama_lokasi}"
    )

    print(
        f"Mulai        : {waktu_mulai}"
    )

    print(
        f"Sampai       : {waktu_akhir}"
    )

    print(
        f"Jumlah data  : {len(data_kualitas_udara)}"
    )

    print(
        f"File JSON    : {output_path}"
    )

    print("=" * 60)


# ============================================================
# 2. TRANSFORM
# Membersihkan data
# ============================================================

def transform_data():

    json_path = (
        "/usr/local/airflow/dags/data/"
        "kualitas_udara.json"
    )

    # --------------------------------------------------------
    # Membaca JSON
    # --------------------------------------------------------

    with open(
        json_path,
        "r",
        encoding="utf-8"
    ) as file:

        data_json = json.load(file)

    latitude = data_json["location"]["latitude"]
    longitude = data_json["location"]["longitude"]

    data = data_json.get("data", [])

    # --------------------------------------------------------
    # Data hasil transformasi
    # --------------------------------------------------------

    data_transform = []

    for row in data:

        waktu = row.get("datetime")

        # Lewati datetime kosong
        if not waktu:
            continue

        # ----------------------------------------------------
        # Nilai polutan
        # ----------------------------------------------------

        nilai_polutan = [
            row.get("pm10"),
            row.get("pm2_5"),
            row.get("carbon_monoxide"),
            row.get("nitrogen_dioxide"),
            row.get("sulphur_dioxide"),
            row.get("ozone")
        ]

        # Jika semua polutan kosong, lewati
        if all(
            value is None
            for value in nilai_polutan
        ):
            continue

        # ----------------------------------------------------
        # Data bersih
        # ----------------------------------------------------

        data_bersih = {
            "datetime": waktu,
            "latitude": latitude,
            "longitude": longitude,
            "pm10": row.get("pm10"),
            "pm2_5": row.get("pm2_5"),
            "carbon_monoxide": row.get("carbon_monoxide"),
            "nitrogen_dioxide": row.get("nitrogen_dioxide"),
            "sulphur_dioxide": row.get("sulphur_dioxide"),
            "ozone": row.get("ozone")
        }

        data_transform.append(data_bersih)

    # --------------------------------------------------------
    # Menghapus duplikasi
    # --------------------------------------------------------

    data_unik = {}

    for row in data_transform:

        key = (
            row["datetime"],
            row["latitude"],
            row["longitude"]
        )

        data_unik[key] = row

    data_transform = list(
        data_unik.values()
    )

    # --------------------------------------------------------
    # Simpan CSV
    # --------------------------------------------------------

    csv_path = (
        "/usr/local/airflow/dags/data/"
        "transformed_kualitas_udara.csv"
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
                "latitude",
                "longitude",
                "pm10",
                "pm2_5",
                "carbon_monoxide",
                "nitrogen_dioxide",
                "sulphur_dioxide",
                "ozone"
            ]
        )

        writer.writeheader()

        writer.writerows(
            data_transform
        )

    # --------------------------------------------------------
    # Log
    # --------------------------------------------------------

    print("=" * 60)
    print("TRANSFORM BERHASIL")
    print("=" * 60)

    print(
        f"Data awal       : {len(data)}"
    )

    print(
        f"Data setelah    : {len(data_transform)}"
    )

    print(
        f"File transform  : {csv_path}"
    )

    print("Transformasi:")
    print("- Menghapus data tanpa datetime")
    print("- Menghapus data yang seluruh polutannya kosong")
    print("- Menghapus data duplikat")

    print("=" * 60)


# ============================================================
# 3. LOAD
# Memasukkan hasil transformasi ke MySQL
# ============================================================

def load_to_mysql():

    # --------------------------------------------------------
    # Airflow Connection
    # --------------------------------------------------------

    mysql_conn = BaseHook.get_connection(
        "mysql-kualitas-udara"
    )

    # --------------------------------------------------------
    # Koneksi MySQL
    # --------------------------------------------------------

    conn = mysql.connector.connect(
        host=mysql_conn.host,
        port=mysql_conn.port or 3306,
        user=mysql_conn.login,
        password=mysql_conn.password,
        database=mysql_conn.schema
    )

    cursor = conn.cursor()

    # --------------------------------------------------------
    # CSV hasil transform
    # --------------------------------------------------------

    csv_path = (
        "/usr/local/airflow/dags/data/"
        "transformed_kualitas_udara.csv"
    )

    # --------------------------------------------------------
    # Query INSERT
    # --------------------------------------------------------

    query = """
        INSERT INTO kualitas_udara (
            datetime,
            latitude,
            longitude,
            pm10,
            pm2_5,
            carbon_monoxide,
            nitrogen_dioxide,
            sulphur_dioxide,
            ozone
        )
        VALUES (
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s
        )
        ON DUPLICATE KEY UPDATE
            pm10 = VALUES(pm10),
            pm2_5 = VALUES(pm2_5),
            carbon_monoxide = VALUES(carbon_monoxide),
            nitrogen_dioxide = VALUES(nitrogen_dioxide),
            sulphur_dioxide = VALUES(sulphur_dioxide),
            ozone = VALUES(ozone)
    """

    jumlah_data = 0

    # --------------------------------------------------------
    # Membaca CSV
    # --------------------------------------------------------

    with open(
        csv_path,
        "r",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            values = (
                row["datetime"],
                float(row["latitude"]),
                float(row["longitude"]),

                float(row["pm10"])
                if row["pm10"]
                else None,

                float(row["pm2_5"])
                if row["pm2_5"]
                else None,

                float(row["carbon_monoxide"])
                if row["carbon_monoxide"]
                else None,

                float(row["nitrogen_dioxide"])
                if row["nitrogen_dioxide"]
                else None,

                float(row["sulphur_dioxide"])
                if row["sulphur_dioxide"]
                else None,

                float(row["ozone"])
                if row["ozone"]
                else None
            )

            cursor.execute(
                query,
                values
            )

            jumlah_data += 1

    # --------------------------------------------------------
    # Commit
    # --------------------------------------------------------

    conn.commit()

    print("=" * 60)
    print("LOAD KE MYSQL BERHASIL")
    print("=" * 60)

    print(
        f"Data diproses : {jumlah_data}"
    )

    print(
        "Database      : kualitas_udara_db"
    )

    print(
        "Tabel         : kualitas_udara"
    )

    print("=" * 60)

    cursor.close()
    conn.close()


# ============================================================
# DAG
# ============================================================

with DAG(
    dag_id="etl_kualitas_udara",

    default_args=default_args,

    # Jalan otomatis setiap jam
    schedule="0 * * * *",

    catchup=False,

    tags=[
        "ETL",
        "Open-Meteo",
        "Air Quality",
        "MySQL",
        "Kendari"
    ],

) as dag:

    # ========================================================
    # CREATE TABLE
    # ========================================================

    create_table = MySqlOperator(
        task_id="create_table_mysql",

        conn_id="mysql-kualitas-udara",

        sql="""
        CREATE TABLE IF NOT EXISTS kualitas_udara (

            id INT AUTO_INCREMENT PRIMARY KEY,

            datetime DATETIME NOT NULL,

            latitude DECIMAL(10, 6),

            longitude DECIMAL(10, 6),

            pm10 FLOAT,

            pm2_5 FLOAT,

            carbon_monoxide FLOAT,

            nitrogen_dioxide FLOAT,

            sulphur_dioxide FLOAT,

            ozone FLOAT,

            UNIQUE KEY unique_waktu_lokasi (
                datetime,
                latitude,
                longitude
            )
        );
        """
    )

    # ========================================================
    # EXTRACT
    # ========================================================

    extract_api_task = PythonOperator(
        task_id="extract_api",

        python_callable=extract_api
    )

    # ========================================================
    # TRANSFORM
    # ========================================================

    transform_data_task = PythonOperator(
        task_id="transform_data",

        python_callable=transform_data
    )

    # ========================================================
    # LOAD
    # ========================================================

    load_to_mysql_task = PythonOperator(
        task_id="load_to_mysql",

        python_callable=load_to_mysql
    )

    # ========================================================
    # URUTAN ETL
    # ========================================================

    (
        create_table
        >> extract_api_task
        >> transform_data_task
        >> load_to_mysql_task
    )