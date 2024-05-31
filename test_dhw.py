import pandas as pd
import numpy as np
from utils import misc
from dhw import domestic_hot_water
from utils import gaussian_occupancy
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import time


df = misc.df_index_timestamp()

occupancy_probabilities = gaussian_occupancy.occupancy_distribution()
occupancy_daily = gaussian_occupancy.defined_time_occupancy(
    occupancy_probabilities)

dhw_daily = domestic_hot_water.dhw_year_day(occupancy_daily)[0]

df["occupancy"] = misc.flatten(occupancy_daily)
df["dhw_litres"] = misc.flatten(dhw_daily)

selected_days = df.loc["2021-01-01":"2021-01-03"]
unique_dates = np.unique(selected_days.index.date)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(80, 40), sharex=True)

# Plot Occupancy
ax1.set_title("Occupancy Profile - 2021-01-01 and 2021-01-03", fontsize=80)
for i, date in enumerate(unique_dates):
    data = selected_days[selected_days.index.date == date]
    ax1.bar(data.index, data["occupancy"], label=str(
        date), alpha=0.7, width=0.03, linewidth=0.5)

ax1.set_ylabel("Occupancy", fontsize=80)
ax1.legend(fontsize=60)

# Plot DHW Draws
for i, date in enumerate(unique_dates):
    data = selected_days[selected_days.index.date == date]
    ax2.bar(data.index, data["dhw_litres"], color=ax1.patches[i].get_facecolor(
    ), alpha=0.7, width=0.03, linewidth=0.5)

ax2.set_xlabel("Time", fontsize=80)
ax2.set_ylabel("DHW Consumption", fontsize=80)

ax1.grid(True, which='both', linestyle=':', linewidth=5, alpha=1)
ax2.grid(True, which='both', linestyle=':', linewidth=5, alpha=1)

ax1.xaxis.set_major_locator(mdates.HourLocator(interval=1))
ax2.xaxis.set_major_locator(mdates.HourLocator(interval=1))

ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

plt.tight_layout()
plt.show()
