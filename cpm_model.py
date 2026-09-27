"""Change-point regression used by the residential HVAC CPM workflow."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import optimize, stats


def piecewise_linear(
    x: np.ndarray,
    heating_change_point: float,
    heating_slope: float,
    cooling_change_point: float,
    cooling_slope: float,
) -> np.ndarray:
    """Evaluate a 3-, 4-, or 5-parameter change-point model."""
    values = np.asarray(x, dtype=float)
    hcp, hsl = heating_change_point, heating_slope
    ccp, csl = cooling_change_point, cooling_slope
    if np.isnan(hcp) and np.isnan(hsl):
        hcp, hsl = ccp, 0.0
    if np.isnan(ccp) and np.isnan(csl):
        ccp, csl = hcp, 0.0
    return np.piecewise(
        values,
        [values < hcp, (values >= hcp) & (values <= ccp), values > ccp],
        [lambda items: hsl * (items - hcp), lambda items: 0.0, lambda items: csl * (items - ccp)],
    )


class InverseModel:
    """Fit an inverse change-point model to daily temperature and HVAC runtime."""

    def __init__(
        self,
        temperature: np.ndarray,
        load: np.ndarray,
        mode: np.ndarray,
        variable_name: str,
        significance_threshold: float = 0.1,
    ) -> None:
        clean = pd.DataFrame({"temperature": temperature, "load": load, "mode": mode}).dropna()
        if len(clean) < 8:
            raise ValueError("At least eight complete daily observations are required.")
        self.temperature = clean["temperature"].to_numpy(dtype=float)
        self.load = clean["load"].to_numpy(dtype=float)
        self.mode = clean["mode"].to_numpy(dtype=float)
        self.variable_name = variable_name
        self.significance_threshold = significance_threshold
        self.cool_only = not np.any(self.mode == 2)
        self.heat_only = not np.any(self.mode == 1)
        self.hsl_insignificant = self.cool_only
        self.csl_insignificant = self.heat_only
        self.has_fit = False

        hcp_percentile, ccp_percentile = (0, 10) if self.cool_only else ((70, 100) if self.heat_only else (60, 65))
        self.hcp, self.ccp = np.percentile(self.temperature, [hcp_percentile, ccp_percentile])
        self.hcp_min = self.hcp
        self.hcp_max = self.ccp
        self.ccp_min = self.hcp
        self.ccp_max = self.ccp
        self.hsl_min, self.hsl_max = -np.inf, 0.0
        self.csl_min, self.csl_max = 0.0, np.inf

    @staticmethod
    def _curve(x: np.ndarray, hcp: float, ccp: float, hsl: float, csl: float) -> np.ndarray:
        return piecewise_linear(x, hcp, hsl, ccp, csl)

    def _fit_once(self) -> None:
        self.p, self.covariance = optimize.curve_fit(
            self._curve,
            self.temperature,
            self.load,
            bounds=(
                [self.hcp_min, self.ccp_min, self.hsl_min, self.csl_min],
                [self.hcp_max, self.ccp_max, self.hsl_max, self.csl_max],
            ),
            maxfev=20000,
        )
        self.hcp, self.ccp, self.hsl, self.csl = self.p
        n = len(self.temperature)
        standard_errors = np.sqrt(np.maximum(np.diag(self.covariance), 0.0) / n)
        self.p_hsl = stats.t.cdf(self.hsl / standard_errors[2], df=max(n - 2, 1)) if standard_errors[2] else 0.0
        self.p_csl = stats.t.sf(self.csl / standard_errors[3], df=max(n - 2, 1)) if standard_errors[3] else 0.0

    def predict(self, temperature: np.ndarray) -> np.ndarray:
        """Predict runtime using the fitted parameters."""
        return piecewise_linear(np.asarray(temperature), *self.model_parameters)

    def rmse(self) -> float:
        residuals = self.load - self._curve(self.temperature, *self.p)
        return float(np.sqrt(np.mean(residuals**2)))

    def r_squared(self, *, adjusted: bool = False) -> float:
        residuals = self.load - self._curve(self.temperature, *self.p)
        ss_res = float(np.sum(residuals**2))
        ss_tot = float(np.sum((self.load - np.mean(self.load)) ** 2))
        if ss_tot == 0:
            return float("nan")
        value = 1 - ss_res / ss_tot
        if adjusted:
            n = len(self.load)
            parameter_count = int(np.count_nonzero(np.isfinite(self.p)))
            if n > parameter_count + 1:
                value = 1 - (1 - value) * (n - 1) / (n - parameter_count - 1)
        return float(value)

    def _optimize_change_point(self, side: str) -> None:
        if side == "heating":
            ranges = np.arange(0, 30, 5) if self.cool_only else (np.arange(80, 100, 5) if self.heat_only else np.arange(20, 90, 5))
        else:
            ranges = np.arange(0, 20, 5) if self.cool_only else (np.arange(70, 100, 5) if self.heat_only else np.arange(10, 80, 5))

        candidates: list[tuple[float, float, float]] = []
        for lower_percentile in ranges:
            lower, upper = np.percentile(self.temperature, [lower_percentile, min(lower_percentile + 5, 100)])
            if side == "heating":
                self.hcp_min, self.hcp_max = lower, upper
            else:
                self.ccp_min, self.ccp_max = lower, upper
            try:
                self._fit_once()
                candidates.append((self.r_squared(), lower, upper))
            except (RuntimeError, ValueError):
                continue
        if not candidates:
            raise RuntimeError(f"Could not optimize the {side} change-point bounds.")
        _, lower, upper = max(candidates, key=lambda item: item[0])
        if side == "heating":
            self.hcp_min, self.hcp_max = lower, upper
        else:
            self.ccp_min, self.ccp_max = lower, upper
        self._fit_once()

    def _optimize_slopes(self) -> None:
        if self.cool_only or not np.isfinite(self.p_hsl) or self.p_hsl >= self.significance_threshold:
            self.hsl_min = -1e-3
            self.hsl_insignificant = True
        if self.heat_only or not np.isfinite(self.p_csl) or self.p_csl >= self.significance_threshold:
            self.csl_max = 1e-3
            self.csl_insignificant = True
        self._fit_once()

    def fit(self, *, minimum_r_squared: float = 0.1) -> "InverseModel":
        """Fit and classify the model, raising when no meaningful fit is available."""
        self._fit_once()
        if not np.isfinite(self.r_squared()) or self.r_squared() < minimum_r_squared:
            raise RuntimeError(f"No meaningful CPM fit was found (R-squared < {minimum_r_squared}).")
        self._optimize_slopes()
        self._optimize_change_point("heating")
        self._optimize_change_point("cooling")
        self._optimize_slopes()

        if self.hsl_insignificant:
            self.hcp, self.hsl = self.ccp, 0.0
        if self.csl_insignificant:
            self.ccp, self.csl = self.hcp, 0.0
        if self.hcp > self.ccp and self.hsl and self.csl:
            common = (self.hsl * self.hcp - self.csl * self.ccp) / (self.hsl - self.csl)
            self.hcp = self.ccp = common

        self.model_type = self._classify_model()
        self.model_parameters = np.array([self.hcp, self.hsl, self.ccp, self.csl], dtype=float)
        self.has_fit = True
        return self

    def _classify_model(self) -> str:
        same_change_point = math.isclose(self.hcp, self.ccp, abs_tol=0.5)
        if self.hsl == 0 and self.csl == 0:
            return "No fit"
        if self.hsl == 0:
            return "3P Cooling"
        if self.csl == 0:
            return "3P Heating"
        return "4P" if same_change_point else "5P"


def fit_cpm(
    daily_data: pd.DataFrame,
    value_column: str = "Runtimes",
    *,
    minimum_r_squared: float = 0.1,
) -> tuple[dict[str, object], pd.DataFrame, InverseModel]:
    """Fit CPM data and return plotting data, a result table, and the model."""
    required = ["Avg_T_ext", value_column, "Mode"]
    missing = [column for column in required if column not in daily_data]
    if missing:
        raise ValueError(f"Daily CPM data is missing required columns: {missing}")
    data = daily_data[required].dropna().rename(columns={"Avg_T_ext": "temp", value_column: "value", "Mode": "mode"})
    model = InverseModel(data["temp"].to_numpy(), data["value"].to_numpy(), data["mode"].to_numpy(), value_column)
    model.fit(minimum_r_squared=minimum_r_squared)

    temperature_plot = np.linspace(data["temp"].min(), data["temp"].max(), 100)
    load_plot = model.predict(temperature_plot)
    result = [*model.model_parameters, model.rmse(), model.r_squared(adjusted=True), model.model_type]
    results = pd.DataFrame(
        {"Result": result},
        index=[
            "Heating Base Temp", "Heating Slope", "Cooling Base Temp", "Cooling Slope",
            "RMSE", "R-squared", "Model Type",
        ],
    )
    plot_data = {"data": data, "temp_plot": temperature_plot, "load_plot": load_plot}
    return plot_data, results, model


def plot_cpm_fit(plot_data: dict[str, object], *, ax=None):
    """Plot daily observations and the fitted CPM curve."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    data = plot_data["data"]
    colors = np.where(data["mode"].to_numpy() == 1, "tab:blue", np.where(data["mode"].to_numpy() == 2, "tab:red", "0.6"))
    ax.scatter(data["temp"], data["value"], alpha=0.35, s=14, c=colors)
    ax.plot(plot_data["temp_plot"], plot_data["load_plot"], color="black", linewidth=2)
    ax.set_xlabel("Outdoor temperature, daily mean (degC)")
    ax.set_ylabel("System runtime (hours/day)")
    ax.set_title("Change-point model fit")
    return ax
