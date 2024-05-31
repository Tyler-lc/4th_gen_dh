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
