import mysql.connector
import sys
import os
import json

    

def create_connection():
    try:
        connection = mysql.connector.connect(
            host='localhost',
            user='DH_project',
            password='H*5mGq@4cSUWDFYKrix9'
        )
        if connection.is_connected():
            print("Connected to MySQL server")
        return connection
    except Error as e:
        print(f"Error: {e}")
        return None

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

# Example data
data = {
    'building_id': 1,
    'building_type': 'SFH1',
    'n_floors': 4,
    'compactness': 0.623345147,
    'volume': 1399.3,
    'gfa': 436,
    'heated_gfa': 350,
    'qgis_specific_HD': 91.45794286,
    'roof_area': 272.09,
    'roof_u_value': 1.017676347,
    'walls_area': 432.7,
    'walls_u_value': 1.418322947,
    'ground_contact_area': 167.41,
    'ground_contact_u_value': 1.128287911,
    'door_area': 1.8,
    'door_u_value': 2.490652577,
    'windows': json.dumps({
        'north': {'area': 13.68418425, 'u_value': 2.31991313, 'shgc': 0.65},
        'northwest': {'area': 0, 'u_value': 0, 'shgc': 0},
        'west': {'area': 0, 'u_value': 0, 'shgc': 0},
        'southwest': {'area': 0, 'u_value': 0, 'shgc': 0},
        'south': {'area': 6.813974751, 'u_value': 2.31991313, 'shgc': 0.65},
        'southeast': {'area': 0, 'u_value': 0, 'shgc': 0},
        'east': {'area': 0, 'u_value': 0, 'shgc': 0},
        'northeast': {'area': 0, 'u_value': 0, 'shgc': 0}
    })
}

if __name__ == "__main__":
    connection = create_connection()
    if connection:
        create_database(connection, 'building_stock')
        connection.database = 'building_stock'
        create_table(connection)
        insert_data(connection, data)
        connection.close()
