import numpy as np
import matplotlib.pyplot as plt
from utils import misc
from utils import gaussian_occupancy
import pandas as pd

# remember this is to create hours! Not Days! If you create days it will not match with the size coming out of
# whole_year_occupancy
total_hours = 8760
test = misc.df_index_timestamp(periods = total_hours, frequency = "1H")

total_days = total_hours//24
#
# timestamps = pd.date_range(start='1/1/2018', periods=8760, freq="1H")  # Adjusted frequency to '1H'
# df = pd.DataFrame(index=timestamps)
#

# linear_space = np.linspace(0, 23, 24)
occupancy_distribution = gaussian_occupancy.occupancy_distribution()
occupancy_daily = gaussian_occupancy.defined_time_occupancy(occupancy_distribution, days=total_days)
test["occupancy"] = misc.flatten(occupancy_daily)


# # Plot the occupancy data for each day as individual bars

# # Get the unique dates for the selected days
selected_days = test.loc["2021-01-01":"2021-01-03"]
unique_dates = np.unique(selected_days.index.date)

fig, ax = plt.subplots()
for i, date in enumerate(unique_dates):
    data = selected_days[selected_days.index.date == date]
    ax.bar(data.index, data["occupancy"], label=str(date), alpha=0.7, width=0.03)

ax.set_xlabel("Time")
ax.set_ylabel("Occupancy")
ax.set_title("Occupancy Profile - 2021-01-01 and 2021-01-03")
# ax.legend()

plt.show()

number_people_dwelling = misc.calculate_number_people(100)