As of 28/05/2024 I think the code is quite optimized now. May need some further little adjustment, but it seems to be running fast and quite precisely.

## Next steps

1. Test the code on other buildings as well, such as a Multi Family House for which we already have the results.
2. Before we do testing, also change the previous project to update the value for that one as well. If we do the comparison with the old code we might find that the results vary a little bit. So update the GELSEP as well.
3. We need to add the occupation in the building in the code as well. We had done this already a couple of generations ago in the code. Check an older project for it.
   1. We need to decide whether this is part of the building class or if we want to call the function from another file. This might make the code easier to read. Maintain all space heating data inside of the building class, while moving other things somewhere else.
   2. SIA 2024 Conditions d’utilisation standard pour l'énergie et les installations du bâtiment,2006. This standard is used in this paper [link](https://www.researchgate.net/publication/312941255_Assessing_the_challenges_of_changing_electricity_demand_profiles_caused_by_evolving_building_stock_and_climatic_conditions_on_distribution_grids)
4. We need to start adding the domestic hot water demand.
   1. We need to decide if it is part of the building class or not. This might be better maybe? although we need to add a toggle for the DHW calculation at this point. It might not always be useful to calculate DHW.
   2. Can we actually call the DHW_calc routine for this? It might be possible since we have also an excel version of it i think
5. We need to find a way to assess the amount of people in each building. This is more important now since we are also calculating DHW and we want to start implementing the occupation.
   1. We can use some Fire code standards for maximum occupancy?
   2. Maybe Ashrae code could help? [link](https://www.ashrae.org/technical-resources/bookstore/standard-90-1)

[This paper](https://www.researchgate.net/publication/276036252_A_Methodology_for_Defining_Electricity_Demand_in_Energy_Simulations_Referred_to_the_Italian_Context) might have some information on occupancy profiles for residential and office types buildings.

[This file](https://github.com/utwente-energy/alpg/blob/master/persons.py) contains some inspiration on how to handle people in the building regarding the DHW and maybe also the temperature of the thermostat.

Another factor to consider is the ability to tell what share of the external walls are shared with another building. This would mean a reduced effective external wall area. It should be possible somehow, not sure how to implement it yet. We already have a TIFF file with the height of the buildings in the entirity of Germany. We calculate the average of the height within the border of each building. This will give us the estimated height of the building. Two adjacient buildings (i.e. sharing at least one wall), will most likely have different heights. knowning the overlap on the top view and the building height, we should be able to calculate the shared wall. This might include also a pitched rooftop. I don't know yet how we are going to the the difference with the shared walls if the building has the pitched roof. Because until now the pitched roof is decided during the building geometries generation. We might need to calculate the shared walls during the geometry creation. So it should be one of the inputs? But it will also require the adjacent building's geometry?

### Copilot's answer on dhw implementation with Building class

Given the complexity of the calculations and the potential for reuse, it would be beneficial to keep the occupancy and domestic hot water demand calculations as separate modules. This would make the code more modular, easier to maintain, and potentially reusable in other parts of the project or in future projects.

You could consider creating a Person class if the characteristics or behaviors of individuals within the building significantly impact the calculations. This class could encapsulate properties like daily water usage, occupancy times, etc.

As for the relationship between the Person and [Building](command:_github.copilot.openSymbolFromReferences?%5B%7B%22%24mid%22%3A1%2C%22path%22%3A%22%2Fc%3A%2FVSCode_python%2F4th_gen_dh%2FBuilding%2FBuilding.py%22%2C%22scheme%22%3A%22file%22%7D%2C%7B%22line%22%3A7%2C%22character%22%3A6%7D%5D "Building/Building.py") classes, it would make sense for a [Building](command:_github.copilot.openSymbolFromReferences?%5B%7B%22%24mid%22%3A1%2C%22path%22%3A%22%2Fc%3A%2FVSCode_python%2F4th_gen_dh%2FBuilding%2FBuilding.py%22%2C%22scheme%22%3A%22file%22%7D%2C%7B%22line%22%3A7%2C%22character%22%3A6%7D%5D "Building/Building.py") instance to contain a list of Person instances. This would reflect the real-world relationship between buildings and people, where a building can contain multiple people, and each person belongs to a single building.

Here's a rough sketch of how you might structure this:

```python

class Person:
   def __init__(self, daily_water_usage, occupancy_times):
      self.daily_water_usage = daily_water_usage
      self.occupancy_times = occupancy_times

class Building:
   def __init__(self, name, building_type, components, outside_temperature, irradiation_data, soil_temp=8, year_start=2015, summer_start=6, summer_end=9):
      self.name = name
      self.building_type = building_type
      self.components = components
      self.outside_temperature = outside_temperature
      self.soil_temp = soil_temp
      self.summer_start = summer_start
      self.summer_end = summer_end
      self.people = []
      # ... rest of your existing code ...

   def add_person(self, person):
      self.people.append(person)

   def calculate_total_daily_water_usage(self):
      return sum(person.daily_water_usage for person in self.people)
```

In this setup, you can easily add people to a building and calculate the total daily water usage for the building. You can also extend the Person and [Building](command:_github.copilot.openSymbolFromReferences?%5B%7B%22%24mid%22%3A1%2C%22path%22%3A%22%2Fc%3A%2FVSCode_python%2F4th_gen_dh%2FBuilding%2FBuilding.py%22%2C%22scheme%22%3A%22file%22%7D%2C%7B%22line%22%3A7%2C%22character%22%3A6%7D%5D "Building/Building.py") classes with additional methods as needed to perform more complex calculations or behaviors.

Remember to update the dhw_profile function in dhw/domestic_hot_water.py and the dhw_input_generator function in utils/misc.py to use the new Person class and its properties.
