import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


pipe_data = pd.read_csv("pipe_costs.csv")

type_of_pipe = "SERIES 2"

fitting_data = pd.DataFrame()
fitting_data["DN"] = pipe_data["DN"] / 1000
fitting_data["paved_cost"] = pipe_data["Paved area/Existing district"]
fitting_data["piping_cost"] = pipe_data[type_of_pipe]


xdata = fitting_data["DN"]
ydata = fitting_data["paved_cost"]


def dig_function(x, fc_dig_st, vc_dig_st, vc_dig_st_ex):
    return fc_dig_st + (vc_dig_st * x) ** vc_dig_st_ex


def piping_function(x, fc_pip, vc_pip, vc_pip_ex):
    return fc_pip + (vc_pip * x) ** vc_pip_ex


# Initial guesses for parameters
p0_dig = [0, 400, 0.5]  # Adjust these based on your expected values
# Bounds for parameters (lower, upper)
bounds_dig = ([0, 0, 0], [np.inf, np.inf, 10])

parameters_dig, covariance_dig = curve_fit(
    dig_function, xdata, ydata, p0=p0_dig, bounds=bounds_dig
)
fit_y = dig_function(xdata, *parameters_dig)
fc_dig = parameters_dig[0]
vc_dig = parameters_dig[1]
vc_dig_ex = parameters_dig[2]

print(f"fc_dig: {fc_dig}")
print(f"vc_dig: {vc_dig}")
print(f"vc_dig_ex: {vc_dig_ex}")

plt.plot(xdata, ydata, "bo", label="Original data")
plt.plot(xdata, fit_y, "r-", label="Fitted curve")
plt.legend()
plt.show()

# now we do the same for the piping costs

xdata = fitting_data["DN"]
ydata = fitting_data["piping_cost"]

# Similar approach for piping function
p0_pip = [20, 400, 0.7]  # Adjust these based on your expected values
bounds_pip = ([0, 0, 0], [np.inf, np.inf, 10])

parameters_pip, covariance_pip = curve_fit(
    piping_function, xdata, ydata, p0=p0_pip, bounds=bounds_pip
)
fit_y = piping_function(xdata, *parameters_pip)
fc_pip = parameters_pip[0]
vc_pip = parameters_pip[1]
vc_pip_ex = parameters_pip[2]

print(f"fc_pip: {fc_pip}")
print(f"vc_pip: {vc_pip}")
print(f"vc_pip_ex: {vc_pip_ex}")

plt.plot(xdata, ydata, "bo", label="Original data")
plt.plot(xdata, fit_y, "r-", label="Fitted curve")
plt.legend()
plt.show()

# now we have the parameters for the dig and piping costs
