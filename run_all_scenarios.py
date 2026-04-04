import subprocess

# List of scripts to run in order
scripts_to_run = [
    "05b_HT_Scenario.py",
    "07_LT_Scenario2.py",
    "08_Booster_Scenario.py",
    "09b_HT_Sens_Analysis.py",
    "09c_LT_Sens_Analysis.py",
    "09d_HT_Booster_Sens_Analysis.py",
    "10a_HT_scenarios_gas_vs_electicity.py",
    "10b_LT_scenarios_gas_vs_electicity.py",
    "10c_HT_Booster_gas_vs_electricity.py",
    "10d_LT_scenarios_gas_vs_electicity_vs_rencosts.py",
    "11_plots_base_cases.py",
    "11b_plot_gas_electr_sensitivity.py",
    "11c_plot_gas_electr_renovations_sensitivity.py",
    "11c_all_types_contour.py",
    "11d_plot_sensitivities_base.py",
    "12_buildingstock_analysis.py",
    "13_DH_parameters.py",
    "14_NFA_calculation.py",
    "15_iwu_renovation_costs_plot.py",
    "16_pumping_losses.py",
    "17_subsidy_calculation.py",
    "18_installed_capacity_summary.py",
    "19_breakeven_at_50pct_reno_subsidy.py",
    "20_heat_demand_profile.py",
    "21_reduction_factor_slopes.py",
    "22_reduction_factor_slopes_table.py",
    "23_contour_slopes.py",
    "24_study_area_map.py",
]

# Run each script in sequence
for script in scripts_to_run:
    print(f"Running {script}...")
    try:
        subprocess.run(["python", script], check=True)
        print(f"Successfully completed {script}")
    except subprocess.CalledProcessError as e:
        print(f"Error running {script}: {e}")
        # break  # Optional: stop if one script fails
    print("-" * 90)

print("All scripts completed")
