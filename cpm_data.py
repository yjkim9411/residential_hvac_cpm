"""Data preparation helpers for daily change-point models."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


EPLUS_COLUMNS = {
    "datetime": "Date/Time",
    "indoor_temperature": "LIVING_UNIT1:Zone Mean Air Temperature [C](TimeStep)",
    "heating_setpoint": "LIVING_UNIT1:Zone Thermostat Heating Setpoint Temperature [C](TimeStep)",
    "cooling_setpoint": "LIVING_UNIT1:Zone Thermostat Cooling Setpoint Temperature [C](TimeStep)",
    "outdoor_temperature": "Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)",
    "dx_heating_runtime": "MAIN DX HEATING COIL_UNIT1:Heating Coil Runtime Fraction [](TimeStep)",
    "supplemental_heating_runtime": "SUPP HEATING COIL_UNIT1:Heating Coil Runtime Fraction [](TimeStep)",
    "fuel_heating_runtime": "MAIN FUEL HEATING COIL_UNIT1:Heating Coil Runtime Fraction [](TimeStep)",
    "cooling_runtime": "DX COOLING COIL_UNIT1:Cooling Coil Runtime Fraction [](TimeStep)",
}


def read_table(path: str | Path) -> pd.DataFrame:
    """Read a CSV or parquet input table."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def split_energyplus_datetime(value: str) -> tuple[int, int, int, int, int]:
    """Parse an EnergyPlus month/day timestamp, including the 24:00 convention."""
    text = str(value).strip()
    rolls_to_next_day = "24:00:00" in text
    if rolls_to_next_day:
        text = text.replace("24:00:00", "00:00:00")
    parsed = datetime.strptime(text, "%m/%d %H:%M:%S")
    if rolls_to_next_day:
        parsed += timedelta(days=1)
    return parsed.year, parsed.month, parsed.day, parsed.hour, parsed.minute


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def assign_eplus_hvac_mode(df: pd.DataFrame) -> pd.DataFrame:
    """Assign 0=off, 1=cooling, and 2=heating from runtime fractions."""
    result = df.copy()
    result["HvacMode"] = 0
    dx_col = EPLUS_COLUMNS["dx_heating_runtime"]
    fuel_col = EPLUS_COLUMNS["fuel_heating_runtime"]
    cool_col = EPLUS_COLUMNS["cooling_runtime"]

    heating = pd.Series(False, index=result.index)
    if dx_col in result:
        heating |= _numeric(result[dx_col]) > 0
    if fuel_col in result:
        heating |= _numeric(result[fuel_col]) > 0
    result.loc[heating, "HvacMode"] = 2
    if cool_col in result:
        result.loc[_numeric(result[cool_col]) > 0, "HvacMode"] = 1
    return result


def calculate_ecobee_heating_runtime(df: pd.DataFrame) -> pd.DataFrame:
    """Combine ecobee compressor and auxiliary heating stages into runtime."""
    result = df.copy()
    for column in ("compHeat1", "compHeat2", "auxHeat1", "auxHeat2"):
        result[column] = _numeric(result[column]) if column in result else 0.0

    has_compressor = result["compHeat1"].sum() > 0
    has_auxiliary = result["auxHeat1"].sum() > 0
    has_second_compressor = result["compHeat2"].sum() > 0
    has_second_auxiliary = result["auxHeat2"].sum() > 0

    comp1_weight = np.where(result["compHeat2"] > 0, 0.7, 1.0)
    comp2_weight = np.where(result["compHeat2"] > 0, 0.3, 0.0)
    aux1_weight = np.where(result["auxHeat2"] > 0, 0.7, 1.0)
    aux2_weight = np.where(result["auxHeat2"] > 0, 0.3, 0.0)
    compressor_runtime = comp1_weight * result["compHeat1"] + comp2_weight * result["compHeat2"]
    auxiliary_runtime = aux1_weight * result["auxHeat1"] + aux2_weight * result["auxHeat2"]

    if has_compressor:
        comp_weight = np.where(result["auxHeat1"] > 0, 0.5, 1.0)
        aux_weight = np.where(result["auxHeat1"] > 0, 0.5, 0.0)
        result["RT_h"] = comp_weight * compressor_runtime + aux_weight * auxiliary_runtime
        system = f"{'2' if has_second_compressor else '1'}-stage HP"
        if has_auxiliary:
            system += f" with {'2' if has_second_auxiliary else '1'}-stage Aux"
        else:
            system += " without Aux"
    elif has_auxiliary:
        result["RT_h"] = auxiliary_runtime
        system = f"{'2' if has_second_auxiliary else '1'}-stage nonHP"
    else:
        result["RT_h"] = 0.0
        system = "No heating"

    result["SYS"] = system
    result["has_heatpump"] = has_compressor
    result["allow_comp_with_aux"] = has_compressor and has_auxiliary
    result["number_heat_stages"] = 2 if (has_second_compressor or has_second_auxiliary) else int(has_compressor or has_auxiliary)
    return result


def calculate_ecobee_cooling_runtime(df: pd.DataFrame) -> pd.DataFrame:
    """Combine ecobee first- and second-stage cooling runtime."""
    result = df.copy()
    for column in ("compCool1", "compCool2"):
        result[column] = _numeric(result[column]) if column in result else 0.0
    first_weight = np.where(result["compCool2"] > 0, 0.7, 1.0)
    second_weight = np.where(result["compCool2"] > 0, 0.3, 0.0)
    result["RT_c"] = first_weight * result["compCool1"] + second_weight * result["compCool2"]
    return result


def _empty_day_frame(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def _split_into_days(df: pd.DataFrame, days: int) -> dict[int, pd.DataFrame]:
    frames: dict[int, pd.DataFrame] = {}
    for day_number in range(1, days + 1):
        frames[day_number] = df.loc[df["day_365"] == day_number].copy()
    return frames


def prepare_eplus_data(
    path: str | Path,
    *,
    days: int = 365,
    interval_seconds: int = 300,
) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    """Prepare EnergyPlus time-step data and return the full table plus daily frames."""
    df = read_table(path).copy()
    required = [
        EPLUS_COLUMNS["datetime"],
        EPLUS_COLUMNS["indoor_temperature"],
        EPLUS_COLUMNS["heating_setpoint"],
        EPLUS_COLUMNS["cooling_setpoint"],
        EPLUS_COLUMNS["outdoor_temperature"],
        EPLUS_COLUMNS["cooling_runtime"],
    ]
    missing = [column for column in required if column not in df]
    if missing:
        raise ValueError(f"EnergyPlus input is missing required columns: {missing}")

    parsed = df[EPLUS_COLUMNS["datetime"]].map(split_energyplus_datetime)
    df[["year", "month", "day", "hour", "minute"]] = pd.DataFrame(parsed.tolist(), index=df.index)
    df = df.drop_duplicates(["year", "month", "day", "hour", "minute"], keep="last").reset_index(drop=True)
    df = assign_eplus_hvac_mode(df)
    dates = pd.to_datetime(dict(year=df["year"], month=df["month"], day=df["day"]))
    df["day_365"] = pd.factorize(dates, sort=False)[0] + 1

    indoor = _numeric(df[EPLUS_COLUMNS["indoor_temperature"]])
    heat_setpoint = _numeric(df[EPLUS_COLUMNS["heating_setpoint"]])
    cool_setpoint = _numeric(df[EPLUS_COLUMNS["cooling_setpoint"]])
    outdoor = _numeric(df[EPLUS_COLUMNS["outdoor_temperature"]])
    df["setpoint"] = np.where(df["HvacMode"] == 1, cool_setpoint, np.where(df["HvacMode"] == 2, heat_setpoint, indoor))
    df["T_int"] = indoor
    df["T_set_h"] = heat_setpoint
    df["T_set_c"] = cool_setpoint
    df["T_ext"] = outdoor
    df["delta_T_int"] = outdoor - indoor
    df["delta_T_set"] = outdoor - df["setpoint"]

    dx = _numeric(df[EPLUS_COLUMNS["dx_heating_runtime"]]) if EPLUS_COLUMNS["dx_heating_runtime"] in df else None
    supplemental = _numeric(df[EPLUS_COLUMNS["supplemental_heating_runtime"]]) if EPLUS_COLUMNS["supplemental_heating_runtime"] in df else None
    fuel = _numeric(df[EPLUS_COLUMNS["fuel_heating_runtime"]]) if EPLUS_COLUMNS["fuel_heating_runtime"] in df else None
    if dx is not None:
        if supplemental is not None:
            comp_weight = np.where(supplemental > 0, 0.5, 1.0)
            supp_weight = np.where(supplemental > 0, 0.5, 0.0)
            heating_fraction = comp_weight * dx + supp_weight * supplemental
        else:
            heating_fraction = dx
    elif fuel is not None:
        heating_fraction = fuel
    else:
        heating_fraction = pd.Series(0.0, index=df.index)

    df["RT_h"] = heating_fraction.clip(0.0, 1.0) * interval_seconds / 3600
    df["RT_c"] = _numeric(df[EPLUS_COLUMNS["cooling_runtime"]]).clip(0.0, 1.0) * interval_seconds / 3600
    df["Time"] = df.groupby("day_365").cumcount() * interval_seconds
    df["Mode"] = df["HvacMode"]
    return df, _split_into_days(df, days)


def prepare_ecobee_data(
    path: str | Path,
    *,
    days: int = 365,
    interval_seconds: int = 300,
    minimum_rows: int | None = 74000,
) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    """Prepare ecobee time-step data and return the full table plus daily frames."""
    df = read_table(path).copy()
    required = [
        "month", "day", "hour", "minute", "HvacMode", "TemperatureExpectedCool",
        "TemperatureExpectedHeat", "Temperature_ctrl", "t_2m:C",
    ]
    missing = [column for column in required if column not in df]
    if missing:
        raise ValueError(f"ecobee input is missing required columns: {missing}")
    if minimum_rows is not None and len(df) < minimum_rows:
        raise ValueError(f"ecobee input has {len(df):,} rows; expected at least {minimum_rows:,}.")

    df = df.drop_duplicates(["month", "day", "hour", "minute"]).reset_index(drop=True)
    df = calculate_ecobee_heating_runtime(df)
    df = calculate_ecobee_cooling_runtime(df)
    df["setpoint"] = np.where(
        df["HvacMode"] == 1,
        df["TemperatureExpectedCool"],
        np.where(df["HvacMode"] == 2, df["TemperatureExpectedHeat"], df["Temperature_ctrl"]),
    )
    date_year = _numeric(df["year"]).astype(int) if "year" in df else pd.Series(2000, index=df.index)
    dates = pd.to_datetime(dict(year=date_year, month=_numeric(df["month"]).astype(int), day=_numeric(df["day"]).astype(int)))
    df["year"] = date_year
    df["day_365"] = pd.factorize(dates, sort=False)[0] + 1
    df["T_int"] = ((df["Temperature_ctrl"] / 10) - 32) * 5 / 9
    df["T_set_h"] = ((df["TemperatureExpectedHeat"] / 10) - 32) * 5 / 9
    df["T_set_c"] = ((df["TemperatureExpectedCool"] / 10) - 32) * 5 / 9
    df["T_ext"] = _numeric(df["t_2m:C"])
    df["delta_T_int"] = df["T_ext"] - df["T_int"]
    df["delta_T_set"] = df["T_ext"] - (((df["setpoint"] / 10) - 32) * 5 / 9)
    df["RT_h"] = _numeric(df["RT_h"]) / 3600
    df["RT_c"] = _numeric(df["RT_c"]) / 3600
    df["Time"] = df.groupby("day_365").cumcount() * interval_seconds
    df["Mode"] = df["HvacMode"]
    return df, _split_into_days(df, days)


def build_cpm_dataset(day_frames: dict[int, pd.DataFrame], *, days: int = 365) -> pd.DataFrame:
    """Aggregate time-step data into the daily table used by the CPM regression."""
    records: list[dict[str, object]] = []
    for day_number in range(1, days + 1):
        frame = day_frames.get(day_number, pd.DataFrame())
        if frame.empty:
            records.append({
                "Year": np.nan, "Month": np.nan, "Day": np.nan, "Date": None,
                "Avg_T_ext": np.nan, "Avg_delta_T_int": np.nan,
                "Avg_delta_T_set": np.nan, "Runtimes": np.nan, "Mode": np.nan,
            })
            continue

        runtime_h = _numeric(frame["RT_h"]).sum()
        runtime_c = _numeric(frame["RT_c"]).sum()
        if runtime_h == runtime_c:
            runtime, mode = np.nan, np.nan
        elif runtime_h > runtime_c:
            runtime, mode = runtime_h, 2
        else:
            runtime, mode = runtime_c, 1

        year = int(frame["year"].iloc[0])
        month = int(frame["month"].iloc[0])
        day = int(frame["day"].iloc[0])
        records.append({
            "Year": year,
            "Month": month,
            "Day": day,
            "Date": f"{year:04d}-{month:02d}-{day:02d}",
            "Avg_T_ext": _numeric(frame["T_ext"]).mean(),
            "Avg_delta_T_int": _numeric(frame["delta_T_int"]).mean(),
            "Avg_delta_T_set": _numeric(frame["delta_T_set"]).mean(),
            "Runtimes": runtime,
            "Mode": mode,
        })
    return pd.DataFrame.from_records(records)
