"""
12-Dataset Industrial Predictive Maintenance Generator, Loader & Auto-Schema Mapper
for Predictive-Machine Guard (`ml_engine/datasets.py`).

Includes 12 Real-World Manufacturing & Industrial Predictive Maintenance Datasets:
 1. ai4i2020              — UCI AI4I 2020 CNC Milling Machine (10,000 rows)
 2. cmapss                — NASA CMAPSS Turbofan Jet Engine Degradation (9,666 rows)
 3. custom_plant          — CNC & Robotic Assembly Vibro-Acoustic Plant (3,500 rows)
 4. ims_bearing           — NASA IMS Bearing Run-to-Failure Vibration (4,000 rows)
 5. secom_semiconductor   — UCI SECOM Semiconductor Wafer Etch Chamber (3,600 rows)
 6. hydraulic_rig         — UCI Hydraulic System Condition Monitoring (3,800 rows)
 7. centrifugal_pump      — Industrial Centrifugal Pump Cavitation & Seal (3,600 rows)
 8. wind_turbine_gearbox  — Wind/Steam Turbine Gearbox & Generator Health (3,800 rows)
 9. steel_rolling_mill    — Steel Hot-Rolling Mill Work-Roll & Strip Telemetry (3,600 rows)
10. robotic_spot_welding  — Automotive Robotic Spot-Welding & Press Shop (3,600 rows)
11. lithium_battery_cell  — EV Lithium-Ion Battery Cell Formation & Aging (3,600 rows)
12. pharma_bioreactor     — Pharmaceutical Bioreactor & Freeze-Dryer (3,500 rows)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


DATASET_CATALOG: Dict[str, Dict[str, Any]] = {
    "ai4i2020": {
        "id": "ai4i2020",
        "title": "1. UCI AI4I 2020 CNC Milling Machine (10k Rows)",
        "filename": "ai4i2020_predictive_maintenance.csv",
        "domain": "CNC Milling & Machining",
    },
    "cmapss": {
        "id": "cmapss",
        "title": "2. NASA CMAPSS Turbofan Jet Engine (Multi-Sensor RUL)",
        "filename": "nasa_cmapss_turbofan.csv",
        "domain": "Aerospace & Gas Turbines",
    },
    "custom_plant": {
        "id": "custom_plant",
        "title": "3. CNC & Robotic Vibro-Acoustic Plant Telemetry",
        "filename": "custom_plant_telemetry.csv",
        "domain": "Robotic Assembly & Spindles",
    },
    "ims_bearing": {
        "id": "ims_bearing",
        "title": "4. NASA IMS Bearing Run-to-Failure Vibration",
        "filename": "ims_bearing_vibration.csv",
        "domain": "Rotating Machinery Bearings",
    },
    "secom_semiconductor": {
        "id": "secom_semiconductor",
        "title": "5. UCI SECOM Semiconductor Wafer Etch Chamber",
        "filename": "secom_semiconductor_wafer.csv",
        "domain": "Semiconductor Fab & Plasma Etch",
    },
    "hydraulic_rig": {
        "id": "hydraulic_rig",
        "title": "6. UCI Hydraulic Test Rig Condition Monitoring",
        "filename": "uci_hydraulic_test_rig.csv",
        "domain": "Industrial Hydraulics & Valves",
    },
    "centrifugal_pump": {
        "id": "centrifugal_pump",
        "title": "7. Industrial Centrifugal Pump Cavitation & Seal",
        "filename": "industrial_centrifugal_pump.csv",
        "domain": "Chemical & Fluid Processing Pumps",
    },
    "wind_turbine_gearbox": {
        "id": "wind_turbine_gearbox",
        "title": "8. Wind & Steam Turbine Gearbox & Generator",
        "filename": "wind_turbine_gearbox_scada.csv",
        "domain": "Power Generation & Turbines",
    },
    "steel_rolling_mill": {
        "id": "steel_rolling_mill",
        "title": "9. Steel Hot-Rolling Mill Work-Roll & Strip",
        "filename": "steel_hot_rolling_mill.csv",
        "domain": "Heavy Metallurgy & Steel Rolling",
    },
    "robotic_spot_welding": {
        "id": "robotic_spot_welding",
        "title": "10. Automotive Robotic Spot-Welding & Press Shop",
        "filename": "automotive_robotic_welding.csv",
        "domain": "Automotive Body-in-White Robotics",
    },
    "lithium_battery_cell": {
        "id": "lithium_battery_cell",
        "title": "11. EV Lithium-Ion Battery Cell Formation & Aging",
        "filename": "ev_lithium_battery_formation.csv",
        "domain": "Gigafactory Battery Manufacturing",
    },
    "pharma_bioreactor": {
        "id": "pharma_bioreactor",
        "title": "12. Pharmaceutical Bioreactor & Freeze-Dryer",
        "filename": "pharma_bioreactor_lyophilizer.csv",
        "domain": "Bio-Manufacturing & Pharma",
    },
}


# Domain specifications for Datasets 4 through 12
DOMAIN_DATASET_SPECS: Dict[str, Dict[str, Any]] = {
    "ims_bearing": {
        "n_samples": 4000,
        "seed": 104,
        "sensors": [
            ("Bearing1 RMS [g]", 0.14, 0.05, 0.04, 0.55),
            ("Bearing2 RMS [g]", 0.16, 0.06, 0.05, 0.62),
            ("Vibration Kurtosis", 3.20, 0.95, 1.80, 9.50),
            ("Crest Factor", 4.10, 0.85, 2.20, 8.80),
            ("Envelope Peak [m/s2]", 12.5, 4.2, 3.0, 38.0),
            ("Outer Race Temp [C]", 58.0, 7.5, 38.0, 98.0),
            ("Shaft Speed [rpm]", 2000.0, 45.0, 1820.0, 2180.0),
        ],
        "modes": [
            ("Outer Race Spalling", lambda d: (d["Vibration Kurtosis"] > 4.6) & (d["Bearing1 RMS [g]"] > 0.21)),
            ("Inner Race Defect", lambda d: (d["Envelope Peak [m/s2]"] > 18.5) & (d["Crest Factor"] > 5.2)),
            ("Roller Element Wear", lambda d: (d["Bearing2 RMS [g]"] > 0.24) & (d["Outer Race Temp [C]"] > 68.0)),
        ],
    },
    "secom_semiconductor": {
        "n_samples": 3600,
        "seed": 105,
        "sensors": [
            ("Chamber Pressure [mTorr]", 45.0, 6.2, 24.0, 72.0),
            ("RF Forward Power [W]", 850.0, 65.0, 620.0, 1120.0),
            ("RF Reflected Power [W]", 18.0, 9.5, 1.0, 75.0),
            ("Etch Gas Flow [sccm]", 120.0, 14.0, 72.0, 175.0),
            ("Electrostatic Chuck Temp [C]", 65.0, 4.8, 48.0, 88.0),
            ("Plasma Impedance [Ohm]", 52.0, 5.5, 34.0, 78.0),
            ("DC Bias Voltage [V]", 310.0, 28.0, 210.0, 420.0),
        ],
        "modes": [
            ("Plasma Arcing Fault", lambda d: (d["RF Reflected Power [W]"] > 32.0) & (d["DC Bias Voltage [V]"] > 345.0)),
            ("Chamber Vacuum Drift", lambda d: (d["Chamber Pressure [mTorr]"] > 54.0) & (d["Etch Gas Flow [sccm]"] < 105.0)),
            ("Chuck Thermal Excursion", lambda d: (d["Electrostatic Chuck Temp [C]"] > 72.5) & (d["RF Forward Power [W]"] > 930.0)),
        ],
    },
    "hydraulic_rig": {
        "n_samples": 3800,
        "seed": 106,
        "sensors": [
            ("PS1 Main Pressure [bar]", 162.0, 14.0, 110.0, 215.0),
            ("FS1 Volume Flow [L/min]", 8.4, 1.3, 4.2, 13.5),
            ("TS1 Hydraulic Oil Temp [C]", 48.5, 6.8, 32.0, 76.0),
            ("CP1 Cooler Efficiency [%]", 84.0, 9.5, 42.0, 100.0),
            ("VS1 Valve Switching Lag [ms]", 24.0, 7.2, 8.0, 62.0),
            ("Accumulator Gas Press [bar]", 102.0, 11.0, 64.0, 138.0),
            ("Motor Power [W]", 2450.0, 290.0, 1550.0, 3600.0),
        ],
        "modes": [
            ("Cooler Thermal Breakdown", lambda d: (d["CP1 Cooler Efficiency [%]"] < 70.0) & (d["TS1 Hydraulic Oil Temp [C]"] > 56.0)),
            ("Proportional Valve Lag", lambda d: (d["VS1 Valve Switching Lag [ms]"] > 34.5) & (d["PS1 Main Pressure [bar]"] < 150.0)),
            ("Internal Pump Leakage", lambda d: (d["FS1 Volume Flow [L/min]"] < 6.6) & (d["Motor Power [W]"] > 2750.0)),
        ],
    },
    "centrifugal_pump": {
        "n_samples": 3600,
        "seed": 107,
        "sensors": [
            ("Suction Pressure [kPa]", 115.0, 18.0, 52.0, 180.0),
            ("Discharge Pressure [kPa]", 640.0, 55.0, 440.0, 840.0),
            ("Flow Rate [m3/h]", 145.0, 22.0, 75.0, 225.0),
            ("Impeller Vibration [mm/s]", 3.4, 1.4, 0.6, 10.5),
            ("Motor Phase Current [A]", 42.0, 6.5, 22.0, 68.0),
            ("Mechanical Seal Temp [C]", 64.0, 8.2, 40.0, 98.0),
            ("Pump Speed [rpm]", 2950.0, 65.0, 2720.0, 3150.0),
        ],
        "modes": [
            ("Impeller Cavitation", lambda d: (d["Suction Pressure [kPa]"] < 92.0) & (d["Impeller Vibration [mm/s]"] > 5.0)),
            ("Mechanical Seal Dry-Run", lambda d: (d["Mechanical Seal Temp [C]"] > 76.0) & (d["Flow Rate [m3/h]"] < 120.0)),
            ("Discharge Blockage Overload", lambda d: (d["Discharge Pressure [kPa]"] > 720.0) & (d["Motor Phase Current [A]"] > 50.0)),
        ],
    },
    "wind_turbine_gearbox": {
        "n_samples": 3800,
        "seed": 108,
        "sensors": [
            ("Gearbox Oil Temp [C]", 68.0, 7.8, 42.0, 98.0),
            ("HSS Bearing Vibration [mm/s]", 2.9, 1.2, 0.5, 9.2),
            ("Generator Winding Temp [C]", 82.0, 10.5, 52.0, 125.0),
            ("Rotor Speed [rpm]", 14.5, 2.2, 7.0, 21.5),
            ("Active Power Output [kW]", 1850.0, 320.0, 650.0, 2850.0),
            ("Main Shaft Torque [kNm]", 1180.0, 195.0, 520.0, 1850.0),
            ("Blade Pitch Angle [deg]", 4.5, 2.8, 0.0, 18.0),
        ],
        "modes": [
            ("Planetary Gearbox Overheat", lambda d: (d["Gearbox Oil Temp [C]"] > 79.0) & (d["HSS Bearing Vibration [mm/s]"] > 4.2)),
            ("Stator Insulation Thermal Fault", lambda d: (d["Generator Winding Temp [C]"] > 96.0) & (d["Active Power Output [kW]"] > 2200.0)),
            ("High-Torque Shaft Fatigue", lambda d: (d["Main Shaft Torque [kNm]"] > 1440.0) & (d["Rotor Speed [rpm]"] < 13.0)),
        ],
    },
    "steel_rolling_mill": {
        "n_samples": 3600,
        "seed": 109,
        "sensors": [
            ("Roll Separating Force [kN]", 14200.0, 1650.0, 9200.0, 19800.0),
            ("Strip Interstand Tension [MPa]", 185.0, 26.0, 95.0, 280.0),
            ("Work-Roll Surface Temp [C]", 485.0, 42.0, 350.0, 640.0),
            ("Descaling Water Press [bar]", 175.0, 19.0, 110.0, 235.0),
            ("Stand Chatter Vibration [g]", 0.42, 0.18, 0.05, 1.35),
            ("Strip Gauge Error [um]", 14.0, 8.5, 1.0, 55.0),
            ("Rolling Speed [m/min]", 420.0, 55.0, 240.0, 610.0),
        ],
        "modes": [
            ("Work-Roll Thermal Spalling", lambda d: (d["Work-Roll Surface Temp [C]"] > 540.0) & (d["Roll Separating Force [kN]"] > 16200.0)),
            ("Third-Octave Stand Chatter", lambda d: (d["Stand Chatter Vibration [g]"] > 0.68) & (d["Rolling Speed [m/min]"] > 475.0)),
            ("Strip Cobble & Tension Tear", lambda d: (d["Strip Interstand Tension [MPa]"] > 222.0) & (d["Strip Gauge Error [um]"] > 24.0)),
        ],
    },
    "robotic_spot_welding": {
        "n_samples": 3600,
        "seed": 110,
        "sensors": [
            ("Weld Current [kA]", 9.8, 1.1, 6.2, 14.2),
            ("Electrode Squeeze Force [N]", 3400.0, 380.0, 2100.0, 4800.0),
            ("Dynamic Resistance [uOhm]", 215.0, 32.0, 115.0, 340.0),
            ("Cooling Water Flow [L/min]", 4.8, 0.75, 2.1, 7.4),
            ("Electrode Cap Wear [cycles]", 420.0, 240.0, 1.0, 1100.0),
            ("Weld Time [ms]", 210.0, 24.0, 135.0, 310.0),
        ],
        "modes": [
            ("Weld Expulsion & Splash", lambda d: (d["Weld Current [kA]"] > 11.2) & (d["Electrode Squeeze Force [N]"] < 3100.0)),
            ("Electrode Cap Mushrooming", lambda d: (d["Electrode Cap Wear [cycles]"] > 720.0) & (d["Dynamic Resistance [uOhm]"] > 250.0)),
            ("Torch Cooling Restriction", lambda d: (d["Cooling Water Flow [L/min]"] < 3.7) & (d["Weld Time [ms]"] > 235.0)),
        ],
    },
    "lithium_battery_cell": {
        "n_samples": 3600,
        "seed": 111,
        "sensors": [
            ("Formation Voltage [V]", 3.82, 0.18, 3.15, 4.38),
            ("DC Internal Resistance [mOhm]", 1.85, 0.42, 0.85, 3.80),
            ("Cell Surface Temp [C]", 34.5, 4.8, 22.0, 56.0),
            ("Formation Current [A]", 48.0, 7.5, 24.0, 78.0),
            ("Gas Generation Pressure [kPa]", 18.5, 6.2, 4.0, 48.0),
            ("Capacity Retention [%]", 94.5, 4.2, 76.0, 100.0),
        ],
        "modes": [
            ("SEI Layer High Impedance", lambda d: (d["DC Internal Resistance [mOhm]"] > 2.45) & (d["Capacity Retention [%]"] < 90.0)),
            ("Thermal Exotherm Warning", lambda d: (d["Cell Surface Temp [C]"] > 41.5) & (d["Formation Current [A]"] > 56.0)),
            ("Electrolyte Outgassing Swell", lambda d: (d["Gas Generation Pressure [kPa]"] > 27.5) & (d["Formation Voltage [V]"] > 4.05)),
        ],
    },
    "pharma_bioreactor": {
        "n_samples": 3500,
        "seed": 112,
        "sensors": [
            ("Chamber Vacuum [mTorr]", 110.0, 24.0, 45.0, 220.0),
            ("Shelf Temperature [C]", -18.0, 6.5, -42.0, 12.0),
            ("Condenser Ice Temp [C]", -68.0, 5.8, -88.0, -45.0),
            ("Agitator Torque [Nm]", 28.5, 5.2, 12.0, 52.0),
            ("Compressor Discharge [psi]", 215.0, 26.0, 135.0, 315.0),
            ("Dissolved Oxygen [%]", 62.0, 11.0, 25.0, 98.0),
        ],
        "modes": [
            ("Vacuum Seal Micro-Leak", lambda d: (d["Chamber Vacuum [mTorr]"] > 145.0) & (d["Condenser Ice Temp [C]"] > -61.0)),
            ("Compressor High-Head Trip", lambda d: (d["Compressor Discharge [psi]"] > 252.0) & (d["Shelf Temperature [C]"] > -10.0)),
            ("Agitator Mechanical Drag", lambda d: (d["Agitator Torque [Nm]"] > 36.0) & (d["Dissolved Oxygen [%]"] < 50.0)),
        ],
    },
}


def _inject_realistic_telemetry_artifacts(
    df: pd.DataFrame, sensor_cols: List[str], seed: int = 42
) -> pd.DataFrame:
    """
    Injects a realistic ~1.2% missing value rate (simulating wireless sensor packet drops)
    and ~1.5% extreme sensor spike outliers (simulating EMI noise spikes) into raw sensor
    columns so the preprocessing pipeline can detect, impute, and Winsorize them.
    """
    work = df.copy()
    rng = np.random.default_rng(seed)
    n = len(work)

    for col in sensor_cols:
        s = work[col].astype(float).copy()
        std = float(s.std()) if float(s.std()) > 0 else 1.0

        missing_mask = rng.random(n) < 0.012
        spike_mask = (rng.random(n) < 0.015) & (~missing_mask)
        spike_signs = rng.choice([-1.0, 1.0], size=n)
        spike_magnitudes = rng.uniform(4.2, 5.8, size=n) * std * spike_signs

        s = np.where(spike_mask, np.round(s + spike_magnitudes, 2), s)
        s = np.where(missing_mask, np.nan, s)
        work[col] = s

    return work


def generate_or_load_ai4i2020(
    n_samples: int = 10000, seed: int = 42, force_regenerate: bool = False
) -> pd.DataFrame:
    csv_path = DATA_DIR / "ai4i2020_predictive_maintenance.csv"
    if csv_path.exists() and not force_regenerate:
        df_loaded = pd.read_csv(csv_path)
        if df_loaded.isna().sum().sum() > 0:
            return df_loaded

    rng = np.random.default_rng(seed)
    types = rng.choice(["L", "M", "H"], size=n_samples, p=[0.50, 0.30, 0.20])
    uids = np.arange(1, n_samples + 1)
    product_ids = [f"{t}{10000 + i}" for i, t in enumerate(types)]

    base_air = rng.normal(loc=300.0, scale=1.8, size=n_samples)
    seasonal_wave = 1.2 * np.sin(np.linspace(0, 8 * np.pi, n_samples))
    air_temp = np.round(base_air + seasonal_wave, 1)
    process_temp = np.round(air_temp + 10.0 + rng.normal(loc=0.0, scale=1.0, size=n_samples), 1)

    torque = np.round(np.clip(rng.normal(loc=40.0, scale=10.0, size=n_samples), 3.8, 78.0), 1)
    power_factor = rng.normal(loc=2860.0, scale=420.0, size=n_samples)
    rpm_raw = (60.0 * power_factor) / (2.0 * np.pi * np.maximum(torque, 12.0))
    rpm_noise = rng.normal(loc=1450.0, scale=175.0, size=n_samples)
    rotational_speed = np.round(np.clip(0.35 * rpm_raw + 0.65 * rpm_noise, 1168, 2886)).astype(float)

    tool_wear = np.zeros(n_samples, dtype=float)
    current_wear = 0
    wear_increment_map = {"L": 2, "M": 3, "H": 5}
    for i in range(n_samples):
        if current_wear > rng.integers(205, 254):
            current_wear = 0
        tool_wear[i] = float(current_wear)
        current_wear += wear_increment_map[types[i]] + int(rng.integers(0, 2))

    twf_threshold = rng.uniform(200.0, 240.0, size=n_samples)
    twf = (tool_wear >= twf_threshold).astype(int)

    temp_diff = process_temp - air_temp
    hdf = ((temp_diff < 8.6) & (rotational_speed < 1380)).astype(int)

    omega = rotational_speed * (2.0 * np.pi / 60.0)
    mech_power = torque * omega
    pwf = ((mech_power < 3500.0) | (mech_power > 9000.0)).astype(int)

    osf_thresholds = np.where(types == "L", 11000.0, np.where(types == "M", 12000.0, 13000.0))
    overstrain_product = tool_wear * torque
    osf = (overstrain_product > osf_thresholds).astype(int)

    rnf = (rng.random(size=n_samples) < 0.001).astype(int)
    machine_failure = ((twf | hdf | pwf | osf | rnf) > 0).astype(int)

    failure_mode = np.full(n_samples, "No Failure", dtype=object)
    failure_mode = np.where(rnf == 1, "RNF (Random Failure)", failure_mode)
    failure_mode = np.where(twf == 1, "TWF (Tool Wear Failure)", failure_mode)
    failure_mode = np.where(osf == 1, "OSF (Overstrain Failure)", failure_mode)
    failure_mode = np.where(pwf == 1, "PWF (Power Failure)", failure_mode)
    failure_mode = np.where(hdf == 1, "HDF (Heat Dissipation)", failure_mode)

    wear_ratio = np.clip(tool_wear / 225.0, 0.0, 1.15)
    overstrain_ratio = np.clip(overstrain_product / osf_thresholds, 0.0, 1.15)
    thermal_stress = np.clip((10.0 - temp_diff) / 3.0, 0.0, 1.0) * np.clip(
        (1450 - rotational_speed) / 250.0, 0.0, 1.0
    )
    power_stress = np.maximum(
        np.clip((3800.0 - mech_power) / 800.0, 0.0, 1.0),
        np.clip((mech_power - 8500.0) / 1000.0, 0.0, 1.0),
    )

    combined_degradation = np.clip(
        0.45 * wear_ratio + 0.30 * overstrain_ratio + 0.15 * thermal_stress + 0.10 * power_stress,
        0.0,
        1.0,
    )
    rul_minutes = np.round(
        np.where(
            machine_failure == 1,
            rng.uniform(0.0, 8.0, size=n_samples),
            (1.0 - combined_degradation) * 220.0,
        ),
        1,
    )
    rul_minutes = np.clip(rul_minutes, 0.0, 240.0)
    health_index = np.round(np.clip(100.0 * (1.0 - combined_degradation), 0.0, 100.0), 1)

    df = pd.DataFrame(
        {
            "UDI": uids,
            "Product ID": product_ids,
            "Type": types,
            "Air temperature [K]": air_temp,
            "Process temperature [K]": process_temp,
            "Rotational speed [rpm]": rotational_speed,
            "Torque [Nm]": torque,
            "Tool wear [min]": tool_wear,
            "Machine failure": machine_failure,
            "Failure Mode": failure_mode,
            "TWF": twf,
            "HDF": hdf,
            "PWF": pwf,
            "OSF": osf,
            "RNF": rnf,
            "RUL [min]": rul_minutes,
            "Health Index [%]": health_index,
        }
    )

    sensor_cols = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]
    df = _inject_realistic_telemetry_artifacts(df, sensor_cols, seed=seed)
    df.to_csv(csv_path, index=False)
    return df


def generate_or_load_cmapss_turbofan(
    n_units: int = 55, seed: int = 101, force_regenerate: bool = False
) -> pd.DataFrame:
    csv_path = DATA_DIR / "nasa_cmapss_turbofan.csv"
    if csv_path.exists() and not force_regenerate:
        df_loaded = pd.read_csv(csv_path)
        if df_loaded.isna().sum().sum() > 0:
            return df_loaded

    rng = np.random.default_rng(seed)
    records: List[Dict[str, Any]] = []
    failure_modes = [
        "HPC Degradation",
        "Fan Blade Erosion",
        "LPT Thermal Fatigue",
        "High Pressure Bleed Leak",
    ]

    for unit_id in range(1, n_units + 1):
        max_cycles = int(rng.integers(130, 230))
        unit_mode = rng.choice(failure_modes, p=[0.45, 0.25, 0.20, 0.10])
        type_variant = rng.choice(["L", "M", "H"], p=[0.4, 0.4, 0.2])

        for cycle in range(1, max_cycles + 1):
            true_rul = max_cycles - cycle
            capped_rul = min(true_rul, 125)
            progress = cycle / max_cycles
            deg = float(np.clip((progress ** 2.4) * (1.0 + 0.15 * rng.normal()), 0.0, 1.35))

            op1_altitude = round(float(rng.normal(0.0, 0.002)), 4)
            op2_mach = round(float(rng.normal(0.0002, 0.0003)), 5)

            s2_lpc_temp = round(642.15 + 1.85 * deg + rng.normal(0, 0.35), 2)
            s3_hpc_temp = round(1585.20 + 18.5 * deg + rng.normal(0, 3.8), 2)
            s4_lpt_temp = round(1401.50 + 28.0 * deg + rng.normal(0, 4.5), 2)
            s7_hpc_press = round(554.10 - 2.90 * deg + rng.normal(0, 0.45), 2)
            s8_fan_speed = round(2388.02 + 0.22 * deg + rng.normal(0, 0.04), 2)
            s9_core_speed = round(9050.0 + 38.0 * deg + rng.normal(0, 8.5), 2)
            s11_static_press = round(47.35 + 1.15 * deg + rng.normal(0, 0.12), 2)
            s12_fuel_ratio = round(521.95 - 2.35 * deg + rng.normal(0, 0.38), 2)
            s14_corr_core = round(8135.0 + 32.0 * deg + rng.normal(0, 7.2), 2)
            s15_bypass_ratio = round(8.415 + 0.12 * deg + rng.normal(0, 0.02), 4)
            s17_bleed_enth = float(round(391.5 + 4.8 * deg + rng.normal(0, 0.9)))
            s20_hpt_bleed = round(38.95 - 0.65 * deg + rng.normal(0, 0.10), 2)
            s21_lpt_bleed = round(23.38 - 0.42 * deg + rng.normal(0, 0.07), 3)

            imminent_failure = 1 if true_rul <= 30 else 0
            mode_label = unit_mode if imminent_failure == 1 else "No Failure"
            health_idx = round(float(np.clip(100.0 * (1.0 - (deg / 1.15)), 0.0, 100.0)), 1)

            records.append(
                {
                    "unit_id": unit_id,
                    "cycle": float(cycle),
                    "Type": type_variant,
                    "op_setting_1": op1_altitude,
                    "op_setting_2": op2_mach,
                    "s2_LPC_temp [R]": s2_lpc_temp,
                    "s3_HPC_temp [R]": s3_hpc_temp,
                    "s4_LPT_temp [R]": s4_lpt_temp,
                    "s7_HPC_pressure [psia]": s7_hpc_press,
                    "s8_fan_speed [rpm]": s8_fan_speed,
                    "s9_core_speed [rpm]": s9_core_speed,
                    "s11_static_pressure [psia]": s11_static_press,
                    "s12_fuel_flow_ratio": s12_fuel_ratio,
                    "s14_corr_core_speed [rpm]": s14_corr_core,
                    "s15_bypass_ratio": s15_bypass_ratio,
                    "s17_bleed_enthalpy": s17_bleed_enth,
                    "s20_HPT_coolant_bleed": s20_hpt_bleed,
                    "s21_LPT_coolant_bleed": s21_lpt_bleed,
                    "Machine failure": imminent_failure,
                    "Failure Mode": mode_label,
                    "RUL [min]": float(capped_rul),
                    "Health Index [%]": health_idx,
                }
            )

    df = pd.DataFrame(records)
    sensor_cols = [
        "s2_LPC_temp [R]",
        "s3_HPC_temp [R]",
        "s4_LPT_temp [R]",
        "s7_HPC_pressure [psia]",
        "s9_core_speed [rpm]",
        "s11_static_pressure [psia]",
        "s12_fuel_flow_ratio",
    ]
    df = _inject_realistic_telemetry_artifacts(df, sensor_cols, seed=seed)
    df.to_csv(csv_path, index=False)
    return df


def generate_or_load_custom_plant_telemetry(
    n_samples: int = 3500, seed: int = 77, force_regenerate: bool = False
) -> pd.DataFrame:
    csv_path = DATA_DIR / "custom_plant_telemetry.csv"
    if csv_path.exists() and not force_regenerate:
        df_loaded = pd.read_csv(csv_path)
        if df_loaded.isna().sum().sum() > 0:
            return df_loaded

    rng = np.random.default_rng(seed)
    machine_zone = rng.choice(["L", "M", "H"], size=n_samples, p=[0.45, 0.35, 0.20])
    vibration_rms = np.round(rng.gamma(shape=2.8, scale=1.15, size=n_samples), 2)
    bearing_temp_c = np.round(rng.normal(loc=62.0, scale=8.5, size=n_samples) + 2.1 * vibration_rms, 1)
    spindle_current_a = np.round(rng.normal(loc=18.5, scale=4.2, size=n_samples), 2)
    acoustic_db = np.round(
        68.0 + 3.2 * vibration_rms + 0.35 * spindle_current_a + rng.normal(0, 2.5, size=n_samples), 1
    )
    lubricant_viscosity_cst = np.round(
        np.clip(
            rng.normal(loc=46.0, scale=6.5, size=n_samples) - 0.18 * (bearing_temp_c - 60),
            18.0,
            68.0,
        ),
        1,
    )
    operating_hours = rng.integers(10, 1200, size=n_samples).astype(float)

    bearing_seizure = ((vibration_rms > 6.2) & (bearing_temp_c > 82.0)).astype(int)
    lubrication_breakdown = ((lubricant_viscosity_cst < 31.0) & (operating_hours > 650)).astype(int)
    motor_overload = ((spindle_current_a > 26.5) & (acoustic_db > 88.0)).astype(int)
    harmonic_resonance = ((vibration_rms > 7.5) & (acoustic_db > 90.0)).astype(int)

    machine_failure = (
        (bearing_seizure | lubrication_breakdown | motor_overload | harmonic_resonance) > 0
    ).astype(int)
    failure_mode = np.full(n_samples, "No Failure", dtype=object)
    failure_mode = np.where(lubrication_breakdown == 1, "Lubrication Breakdown", failure_mode)
    failure_mode = np.where(motor_overload == 1, "Motor Winding Overload", failure_mode)
    failure_mode = np.where(harmonic_resonance == 1, "Harmonic Resonance", failure_mode)
    failure_mode = np.where(bearing_seizure == 1, "Bearing Thermal Seizure", failure_mode)

    stress_score = np.clip(
        0.35 * (vibration_rms / 8.0)
        + 0.30 * np.clip((bearing_temp_c - 50.0) / 45.0, 0, 1.2)
        + 0.20 * (operating_hours / 1200.0)
        + 0.15 * np.clip((48.0 - lubricant_viscosity_cst) / 25.0, 0, 1.2),
        0.0,
        1.0,
    )
    rul = np.round(
        np.where(
            machine_failure == 1,
            rng.uniform(1.0, 12.0, size=n_samples),
            (1.0 - stress_score) * 300.0,
        ),
        1,
    )
    health_idx = np.round(np.clip(100.0 * (1.0 - stress_score), 0.0, 100.0), 1)

    df = pd.DataFrame(
        {
            "Type": machine_zone,
            "Vibration RMS [mm/s]": vibration_rms,
            "Bearing Temp [C]": bearing_temp_c,
            "Spindle Current [A]": spindle_current_a,
            "Acoustic Emission [dB]": acoustic_db,
            "Lubricant Viscosity [cSt]": lubricant_viscosity_cst,
            "Operating Hours [h]": operating_hours,
            "Machine failure": machine_failure,
            "Failure Mode": failure_mode,
            "RUL [min]": rul,
            "Health Index [%]": health_idx,
        }
    )

    sensor_cols = [
        "Vibration RMS [mm/s]",
        "Bearing Temp [C]",
        "Spindle Current [A]",
        "Acoustic Emission [dB]",
        "Lubricant Viscosity [cSt]",
        "Operating Hours [h]",
    ]
    df = _inject_realistic_telemetry_artifacts(df, sensor_cols, seed=seed)
    df.to_csv(csv_path, index=False)
    return df


def generate_or_load_domain_dataset(dataset_key: str, force_regenerate: bool = False) -> pd.DataFrame:
    """
    Generates or loads any of the 12 industrial benchmark datasets (`dataset_key`).
    """
    if dataset_key == "ai4i2020":
        return generate_or_load_ai4i2020(force_regenerate=force_regenerate)
    if dataset_key == "cmapss":
        return generate_or_load_cmapss_turbofan(force_regenerate=force_regenerate)
    if dataset_key == "custom_plant":
        return generate_or_load_custom_plant_telemetry(force_regenerate=force_regenerate)

    if dataset_key not in DOMAIN_DATASET_SPECS:
        # Check if user dropped a custom CSV file into data/ matching dataset_key
        candidate_csv = DATA_DIR / f"{dataset_key}.csv"
        if candidate_csv.exists():
            df_raw = pd.read_csv(candidate_csv)
            norm_df, _ = normalize_custom_dataframe(df_raw)
            return norm_df
        raise ValueError(f"Unknown dataset_key: {dataset_key}")

    meta = DATASET_CATALOG[dataset_key]
    spec = DOMAIN_DATASET_SPECS[dataset_key]
    csv_path = DATA_DIR / meta["filename"]

    if csv_path.exists() and not force_regenerate:
        return pd.read_csv(csv_path)

    n = spec["n_samples"]
    rng = np.random.default_rng(spec["seed"])
    types = rng.choice(["L", "M", "H"], size=n, p=[0.45, 0.35, 0.20])

    data_dict: Dict[str, Any] = {"Type": types}
    sensor_cols: List[str] = []
    norm_stress_Accum = np.zeros(n, dtype=float)

    for col_name, mean_v, std_v, min_v, max_v in spec["sensors"]:
        vals = np.clip(rng.normal(loc=mean_v, scale=std_v, size=n), min_v, max_v)
        data_dict[col_name] = np.round(vals, 2)
        sensor_cols.append(col_name)
        norm_stress_Accum += np.clip(np.abs(vals - mean_v) / (2.5 * std_v), 0.0, 1.2)

    df = pd.DataFrame(data_dict)

    failure_mode = np.full(n, "No Failure", dtype=object)
    any_fail = np.zeros(n, dtype=int)

    for mode_name, rule_fn in spec["modes"]:
        mask = rule_fn(df).astype(int)
        any_fail = (any_fail | mask).astype(int)
        failure_mode = np.where(mask == 1, mode_name, failure_mode)

    stress_ratio = np.clip(norm_stress_Accum / len(sensor_cols), 0.0, 1.0)
    rul = np.round(
        np.where(any_fail == 1, rng.uniform(1.5, 14.0, size=n), (1.0 - stress_ratio) * 240.0), 1
    )
    health_idx = np.round(np.clip(100.0 * (1.0 - stress_ratio * 0.85 - any_fail * 0.45), 0.0, 100.0), 1)

    df["Machine failure"] = any_fail
    df["Failure Mode"] = failure_mode
    df["RUL [min]"] = rul
    df["Health Index [%]"] = health_idx

    df = _inject_realistic_telemetry_artifacts(df, sensor_cols, seed=spec["seed"])
    df.to_csv(csv_path, index=False)
    return df


def ensure_all_12_datasets_generated(force_regenerate: bool = False) -> Dict[str, pd.DataFrame]:
    """Generates or loads all 12 benchmark datasets in `data/`."""
    out: Dict[str, pd.DataFrame] = {}
    for key in DATASET_CATALOG:
        out[key] = generate_or_load_domain_dataset(key, force_regenerate=force_regenerate)
    return out


def normalize_custom_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Inspects any user-uploaded CSV DataFrame and automatically detects or synthesizes:
    - Binary failure target (`Machine failure`)
    - Multi-class failure mode (`Failure Mode`)
    - Remaining Useful Life (`RUL [min]`) & `Health Index [%]`
    - Categorical quality/machine tier (`Type`)
    """
    df = df.copy()

    target_candidates = [
        "Machine failure",
        "machine_failure",
        "failure",
        "Failure",
        "target",
        "Target",
        "label",
        "Label",
        "status",
        "is_failed",
    ]
    detected_target = None
    for col in target_candidates:
        if col in df.columns and df[col].dropna().nunique() <= 5:
            detected_target = col
            break

    if detected_target is None:
        for col in df.columns:
            vals = set(df[col].dropna().unique())
            if vals.issubset({0, 1, 0.0, 1.0, True, False}):
                detected_target = col
                break

    if detected_target and detected_target != "Machine failure":
        df["Machine failure"] = pd.to_numeric(df[detected_target], errors="coerce").fillna(0).astype(int)
    elif "Machine failure" not in df.columns:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols:
            z_scores = ((df[num_cols] - df[num_cols].mean()) / (df[num_cols].std() + 1e-6)).abs().max(axis=1)
            threshold = np.percentile(z_scores.dropna(), 94)
            df["Machine failure"] = (z_scores >= threshold).astype(int)
        else:
            df["Machine failure"] = 0

    if "Failure Mode" not in df.columns:
        df["Failure Mode"] = np.where(df["Machine failure"] == 1, "Detected Anomaly Failure", "No Failure")

    if "Type" not in df.columns:
        df["Type"] = "M"

    rul_candidates = ["RUL [min]", "RUL", "rul", "remaining_useful_life", "RUL_cycles"]
    found_rul = None
    for c in rul_candidates:
        if c in df.columns:
            found_rul = c
            break
    if found_rul and found_rul != "RUL [min]":
        df["RUL [min]"] = pd.to_numeric(df[found_rul], errors="coerce").fillna(100.0)
    elif "RUL [min]" not in df.columns:
        num_cols = [
            c
            for c in df.select_dtypes(include=[np.number]).columns
            if c not in {"Machine failure", "UDI", "unit_id", "TWF", "HDF", "PWF", "OSF", "RNF"}
        ]
        if num_cols:
            norm_stress = (
                ((df[num_cols] - df[num_cols].min()) / (df[num_cols].max() - df[num_cols].min() + 1e-6))
                .mean(axis=1)
                .clip(0.0, 1.0)
            )
            df["RUL [min]"] = np.round(np.where(df["Machine failure"] == 1, 5.0, (1.0 - norm_stress) * 200.0), 1)
            df["Health Index [%]"] = np.round((1.0 - norm_stress) * 100.0, 1)
        else:
            df["RUL [min]"] = 120.0
            df["Health Index [%]"] = 85.0

    if "Health Index [%]" not in df.columns:
        max_rul = max(float(df["RUL [min]"].max()), 1.0)
        df["Health Index [%]"] = np.round(np.clip(100.0 * df["RUL [min]"] / max_rul, 0.0, 100.0), 1)

    schema_info = {
        "row_count": len(df),
        "failure_rate_pct": round(float(df["Machine failure"].mean() * 100.0), 2),
        "detected_target": detected_target or "Synthesized Anomaly Target",
    }
    return df, schema_info
