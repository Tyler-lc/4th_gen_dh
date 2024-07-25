from joblib import Parallel, delayed
import time
from Person.Person import Person
from tqdm import tqdm


def create_person(building_id):
    person = Person(building_id=building_id, person_id=f"Person_{building_id}")
    dhw_profile = person.dhw_profile()  # Run the dhw_profile method
    return person


def main():
    start_time = time.time()
    num_people = 100
    num_cores = -1  # Use all available cores
    chunk_size = 8  # Can be adjusted as needed

    with tqdm(total=num_people, desc="Creating Person instances") as pbar:
        results = Parallel(n_jobs=num_cores, backend="multiprocessing")(
            delayed(create_person)(i) for i in range(num_people)
        )
        pbar.update(len(results))

    end_time = time.time()
    print(
        f"Time taken to create {num_people} Person instances: {end_time - start_time:.2f} seconds"
    )


if __name__ == "__main__":
    main()
