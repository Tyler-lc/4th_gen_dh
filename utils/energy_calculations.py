def energy_dhw(dhw_profile, cold_water_T, target_T):
    cp_water = 4.186  # kJ/kgK heat capacity of water
    energy_demand_dhw = cp_water * dhw_profile * (target_T - cold_water_T)
    return energy_demand_dhw

