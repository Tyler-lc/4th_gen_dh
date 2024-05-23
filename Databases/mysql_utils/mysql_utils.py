from sqlalchemy import create_engine
import pandas as pd
import mysql.connector
from mysql.connector import Error
from urllib.parse import quote_plus

def create_connection():
    try:
        connection = mysql.connector.connect(
            host='localhost',
            user='DH_project',
            password='H*5mGq@4cSUWDFYKrix9',
            database='building_stock'  # Default database
        )
        if connection.is_connected():
            print("Successfully connected to MySQL")
        return connection
    except Error as e:
        print(f"Error: {e}")
        return None

def fetch_data(building_id, db_name: str = 'building_stock'):
    query = f"SELECT * FROM buildings WHERE building_id = {building_id}"
    connection = create_connection()
    if connection is None:
        return None

    user = 'DH_project'
    password = 'H*5mGq@4cSUWDFYKrix9'
    host = 'localhost'
    database = db_name

    
    try:
        # Ensure the password is correctly encoded in the URL
        password_quoted = quote_plus(password)

        # Create SQLAlchemy engine with properly formatted connection string
        engine = create_engine(f'mysql+mysqlconnector://{user}:{password_quoted}@{host}/{database}')
        
        # Read SQL query into a DataFrame
        df = pd.read_sql(query, engine)
        return df
    except Exception as e:
        print(f"Error: {e}")
        return None
    finally:
        if connection.is_connected():
            connection.close()



def create_database(connection, database_name):
    try:
        cursor = connection.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {database_name}")
        cursor.close()
        print(f"Database {database_name} created or already exists.")
    except Error as e:
        print(f"Error: {e}")


def create_table(connection):
    try:
        cursor = connection.cursor()
        create_table_query = """
        CREATE TABLE IF NOT EXISTS buildings (
            building_id INT PRIMARY KEY,
            building_type VARCHAR(50),
            n_floors INT,
            compactness FLOAT,
            volume FLOAT,
            gfa FLOAT,
            heated_gfa FLOAT,
            qgis_specific_HD FLOAT,
            roof_area FLOAT,
            roof_u_value FLOAT,
            walls_area FLOAT,
            walls_u_value FLOAT,
            ground_contact_area FLOAT,
            ground_contact_u_value FLOAT,
            door_area FLOAT,
            door_u_value FLOAT,
            windows JSON
        );
        """
        cursor.execute(create_table_query)
        connection.commit()
        cursor.close()
        print("Table created or already exists.")
    except Error as e:
        print(f"Error: {e}")

def insert_data(connection, data):
    try:
        cursor = connection.cursor()
        insert_query = """
        INSERT INTO buildings 
        (building_id, building_type, n_floors, compactness, volume, gfa, heated_gfa, qgis_specific_HD, roof_area, roof_u_value, walls_area, walls_u_value, ground_contact_area, ground_contact_u_value, door_area, door_u_value, windows)
        VALUES (%(building_id)s, %(building_type)s, %(n_floors)s, %(compactness)s, %(volume)s, %(gfa)s, %(heated_gfa)s, %(qgis_specific_HD)s, %(roof_area)s, %(roof_u_value)s, %(walls_area)s, %(walls_u_value)s, %(ground_contact_area)s, %(ground_contact_u_value)s, %(door_area)s, %(door_u_value)s, %(windows)s)
        """
        cursor.execute(insert_query, data)
        connection.commit()
        cursor.close()
        print("Data inserted successfully.")
    except Error as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    building_id = 1  # Example building ID
    data = fetch_data(building_id)
    if data is not None:
        print(data)
    else:
        print("Failed to fetch data.")
