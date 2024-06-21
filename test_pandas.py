import pandas as pd
import numpy as np

hourly = pd.date_range("2019-01-01", periods=72, freq="H")
daily = pd.date_range("2019-01-01", periods=3, freq="D")
