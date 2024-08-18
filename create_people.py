import os
import sys
import pandas as pd
import numpy as np
import logging
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from Person.Person import Person

# Set up logging
log_file = "building_analysis.log"
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logging.info("Script started.")

# Ensure the parent directory is in the Python path
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Load the building data
buildings_path = "building_analysis/buildingstock/buildingstock.parquet"
buildings_data = pd.read_parquet(buildings_path)

# Filter residential buildings and calculate the number of people per building
res_mask = buildings_data["building_usage"].isin(["sfh", "mfh", "ab", "th"])
total_GFA = buildings_data[res_mask]["GFA"].sum()
maximum_people = 9500
people_per_GFA = maximum_people / total_GFA
buildings_data.loc[res_mask, "n_people"] = round(buildings_data["GFA"] * people_per_GFA)
buildings_data["n_people"] = (
    buildings_data["n_people"].fillna(0).infer_objects(copy=False)
)
buildings_data["n_people"] = buildings_data["n_people"].astype(int)


def process_building(building):
    try:
        fid = building["fid"]
        full_id = building["full_id"]
        osmid = building["osm_id"]
        n_people = building["n_people"]
        logging.info(f"Processing Building {full_id} with {n_people} people.")

        for people in range(n_people):
            logging.info(f"Analyzing person {people} in building {full_id}.")
            person = Person(full_id, people)
            dhw_data = person.dhw_profile()
            os.makedirs("building_analysis/dhw_profiles", exist_ok=True)
            dhw_data.to_csv(f"building_analysis/dhw_profiles/{full_id}_{people}.csv")

        return (full_id, n_people, "success")
    except Exception as e:
        logging.error(f"Error processing building {building['full_id']}: {str(e)}")
        return (building["full_id"], 0, "error")


if __name__ == "__main__":
    buildings_to_process = buildings_data[res_mask].to_dict("records")

    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(process_building, building): building
            for building in buildings_to_process
        }

        with tqdm(total=len(futures)) as pbar:
            successful_buildings = 0
            processed_people = 0
            for future in as_completed(futures):
                result = future.result()
                pbar.update(1)

                if result[2] == "success":
                    successful_buildings += 1
                    processed_people += result[1]

            logging.info(
                f"Total buildings processed successfully: {successful_buildings}"
            )
            logging.info(f"Total people processed: {processed_people}")

    logging.info("Script completed.")
