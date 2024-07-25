import dask
import dask.bag as db
from dask.distributed import Client, progress
import time
from Person.Person import Person


def create_person(building_id):
    person = Person(building_id=building_id, person_id=f"Person_{building_id}")
    dhw_profile = person.dhw_profile()  # Run the dhw_profile method
    return person


def main():
    client = Client(n_workers=16)  # Adjust the number of workers based on your CPU
    num_people = 100

    start_time = time.time()
    bag = db.from_sequence(range(num_people), npartitions=4)
    results = bag.map(create_person).compute()
    end_time = time.time()

    print(
        f"Time taken to create {num_people} Person instances: {end_time - start_time:.2f} seconds"
    )
    client.close()


if __name__ == "__main__":
    main()
