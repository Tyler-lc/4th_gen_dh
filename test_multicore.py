import time
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from Person.Person import Person
from tqdm import tqdm
import os


def create_person(building_id):
    person = Person(building_id=building_id, name=f"Person_{building_id}")
    dhw_profile = person.dhw_profile()  # Run the dhw_profile method
    return person


def main():
    start_time = time.time()
    num_people = 1000
    results = []
    max_workers = os.cpu_count()  # Utilize all available CPU cores

    chunk_size = 64  # Change this value to the desired chunk size

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i in range(0, num_people, chunk_size):  # Submit tasks in smaller chunks
            chunk_futures = [
                executor.submit(create_person, j)
                for j in range(i, min(i + chunk_size, num_people))
            ]
            futures.extend(chunk_futures)
        with tqdm(total=num_people, desc="Creating Person instances") as pbar:
            for future in as_completed(futures):
                results.append(future.result())
                pbar.update(1)  # Update the progress bar

    end_time = time.time()
    print(
        f"Time taken to create {num_people} Person instances: {end_time - start_time:.2f} seconds"
    )


if __name__ == "__main__":
    main()
