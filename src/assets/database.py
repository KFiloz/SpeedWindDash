import psycopg2
import pandas as pd

class DatabaseManager:
    def __init__(self, user, password, host, port, database):
        self.connection_details = {
            "user": user,
            "password": password,
            "host": host,
            "port": port,
            "database": database
        }
        self.connection = None

    def __enter__(self):
        self.connection = psycopg2.connect(**self.connection_details)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.connection:
            self.connection.close()

    def fetch_data(self, query):
        with self.connection.cursor() as cursor:
            cursor.execute(query)
            columns = [desc[0] for desc in cursor.description]
            data = cursor.fetchall()
            return pd.DataFrame(data, columns=columns)