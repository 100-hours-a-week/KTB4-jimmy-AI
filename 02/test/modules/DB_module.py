import pymysql
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../.env"))

def get_connection():
    return pymysql.connect(
        host="localhost",
        user="root",
        password=os.getenv("DB_PASSWORD"),
        database="test_db",
        cursorclass=pymysql.cursors.DictCursor
    )