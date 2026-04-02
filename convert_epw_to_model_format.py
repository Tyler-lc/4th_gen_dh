"""Convert an EnergyPlus EPW file to the irradiation CSV format expected by the
quasi-steady-state building energy demand model.

The model expects hourly irradiation on vertical surfaces (tilt=90°) at four
cardinal orientations (south, east, north, west) in kWh/m², plus air
temperature (T2m) in °C.

The EPW provides GHI, DNI, DHI on the horizontal plane.  pvlib is used to:
  1. compute solar position for each hour,
  2. transpose irradiance onto each vertical facade (Perez model for diffuse).

Output is written in the same layout as the PVGIS-derived CSV the model
currently reads.
"""

import pandas as pd
import pvlib
from pathlib import Path

# --- configuration -----------------------------------------------------------
LAT = 50.05
LON = 8.60
ALTITUDE = 113  # metres, from the EPW header

EPW_PATH = Path("irradiation_data/DEU_Frankfurt.am.Main.106370_IWEC/"
                "DEU_Frankfurt.am.Main.106370_IWEC.epw")

OUT_DIR = Path("irradiation_data/Frankfurt_Griesheim_Mitte_IWEC_TMY")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "Frankfurt_Griesheim_Mitte_IWEC_TMY_irradiation_data.csv"

# facade orientations: pvlib uses 0°=north, 90°=east, 180°=south, 270°=west
FACADES = {
    "south": 180,
    "east": 90,
    "north": 0,
    "west": 270,
}
SURFACE_TILT = 90  # vertical wall

# --- read EPW ----------------------------------------------------------------
epw_data, epw_meta = pvlib.iotools.read_epw(EPW_PATH)

# pvlib.iotools.read_epw returns a DataFrame indexed by a DatetimeIndex
# with columns including 'ghi', 'dni', 'dhi', 'temp_air', etc.
ghi = epw_data["ghi"]
dni = epw_data["dni"]
dhi = epw_data["dhi"]
temp_air = epw_data["temp_air"]

# --- solar position ----------------------------------------------------------
solar_pos = pvlib.solarposition.get_solarposition(
    time=epw_data.index,
    latitude=LAT,
    longitude=LON,
    altitude=ALTITUDE,
    temperature=temp_air,
)

# --- transpose irradiance to each vertical facade ---------------------------
result = pd.DataFrame(index=range(len(epw_data)))

for name, azimuth in FACADES.items():
    total_irrad = pvlib.irradiance.get_total_irradiance(
        surface_tilt=SURFACE_TILT,
        surface_azimuth=azimuth,
        solar_zenith=solar_pos["apparent_zenith"],
        solar_azimuth=solar_pos["azimuth"],
        dni=dni,
        ghi=ghi,
        dhi=dhi,
        model="perez",
        dni_extra=pvlib.irradiance.get_extra_radiation(epw_data.index),
        airmass=pvlib.atmosphere.get_relative_airmass(
            solar_pos["apparent_zenith"]
        ),
    )
    # poa_global is total plane-of-array irradiance in W/m²
    # PVGIS data is also in W/m² (mislabeled as kWh/m² in the column header)
    # so we keep the same units for consistency with the model
    irrad_wh = total_irrad["poa_global"].fillna(0.0).clip(lower=0)
    result[f"{name} G(i) [kWh/m2]"] = irrad_wh.values

result["T2m"] = temp_air.values

# --- write output ------------------------------------------------------------
result.to_csv(OUT_FILE)
print(f"Written {len(result)} rows to {OUT_FILE}")
print(f"\nSample (first 5 rows):\n{result.head()}")
print(f"\nAnnual totals (kWh/m²):")
for col in result.columns:
    if "G(i)" in col:
        print(f"  {col}: {result[col].sum():.1f}")
print(f"  Mean T2m: {result['T2m'].mean():.1f} °C")
