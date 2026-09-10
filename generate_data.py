"""Synthetic data generator for the Pakistan Multi-Hazard Risk Analyzer.

Generates a physically-plausible, reproducible synthetic dataset of 150
Pakistani districts with flood, heatwave, and seismic risk labels derived
from correlated geophysical features (not independently randomised labels).

This is SYNTHETIC data built for a portfolio project. It approximates real
geography (province-conditioned elevation/coordinates, real bounding box)
but district-level values are simulated, not measured. Label this clearly
in any README / CV description.

Usage:
    python generate_data.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("generate_data")

SEED: int = 42
N_DISTRICTS: int = 150
OUTPUT_PATH: Path = Path("data/pakistan_districts.csv")

# Pakistan's real geographic bounding box (approximate land extent).
LAT_MIN, LAT_MAX = 23.6, 37.0
LON_MIN, LON_MAX = 61.0, 77.0

# Real district name pools per province, used cyclically with a numeric
# suffix when a province's pool is exhausted (keeps 150 rows unique while
# staying anchored to real place names where possible).
PROVINCE_DISTRICTS: dict[str, list[str]] = {
    "Punjab": [
        "Lahore", "Faisalabad", "Rawalpindi", "Multan", "Gujranwala",
        "Sialkot", "Bahawalpur", "Sargodha", "Sheikhupura", "Rahim Yar Khan",
        "Jhang", "Dera Ghazi Khan", "Gujrat", "Kasur", "Sahiwal",
        "Okara", "Muzaffargarh", "Vehari", "Chiniot", "Hafizabad",
        "Nankana Sahib", "Toba Tek Singh", "Layyah", "Khanewal", "Mianwali",
        "Bhakkar", "Jhelum", "Attock", "Narowal", "Pakpattan",
    ],
    "Sindh": [
        "Karachi", "Hyderabad", "Sukkur", "Larkana", "Nawabshah",
        "Mirpur Khas", "Jacobabad", "Shikarpur", "Dadu", "Thatta",
        "Badin", "Tando Allahyar", "Ghotki", "Khairpur", "Sanghar",
        "Umerkot", "Tharparkar", "Jamshoro", "Matiari", "Kashmore",
        "Kambar Shahdadkot", "Sujawal", "Tando Muhammad Khan",
    ],
    "Khyber Pakhtunkhwa": [
        "Peshawar", "Mardan", "Abbottabad", "Swat", "Kohat",
        "Dera Ismail Khan", "Bannu", "Mansehra", "Nowshera", "Charsadda",
        "Swabi", "Haripur", "Chitral", "Buner", "Shangla",
        "Tank", "Lakki Marwat", "Karak", "Hangu", "Malakand",
        "Battagram", "Torghar", "Upper Dir", "Lower Dir",
    ],
    "Balochistan": [
        "Quetta", "Gwadar", "Turbat", "Khuzdar", "Sibi",
        "Zhob", "Chaman", "Loralai", "Dera Bugti", "Kalat",
        "Nushki", "Panjgur", "Mastung", "Kharan", "Lasbela",
        "Pishin", "Killa Saifullah", "Jaffarabad", "Naseerabad",
        "Barkhan", "Musakhel",
    ],
    "Gilgit-Baltistan": [
        "Gilgit", "Skardu", "Hunza", "Ghanche", "Ghizer",
        "Nagar", "Shigar", "Astore", "Diamer", "Kharmang",
    ],
    "Azad Jammu & Kashmir": [
        "Muzaffarabad", "Mirpur", "Kotli", "Bagh", "Rawalakot",
        "Bhimber", "Hattian Bala", "Sudhanoti", "Haveli", "Neelum",
    ],
    "Islamabad Capital Territory": ["Islamabad"],
}

# Approximate province-level geographic anchors used to condition sampling
# so lat/lon/elevation stay internally consistent (e.g. GB districts land
# in the north at high elevation, not randomly in the Sindh plains).
PROVINCE_PROFILE: dict[str, dict[str, tuple[float, float]]] = {
    # (mean, std) per attribute, used as Gaussian sampling centres.
    "Punjab": {
        "lat": (31.0, 1.6), "lon": (72.5, 1.8),
        "elevation": (180, 120), "rainfall": (450, 180),
        "river_proximity": (25, 20), "pop_density": (700, 350),
        "temp_proxy": (34, 3),
    },
    "Sindh": {
        "lat": (26.5, 1.5), "lon": (68.3, 1.5),
        "elevation": (60, 45), "rainfall": (180, 90),
        "river_proximity": (18, 15), "pop_density": (600, 500),
        "temp_proxy": (38, 3),
    },
    "Khyber Pakhtunkhwa": {
        "lat": (34.2, 1.3), "lon": (72.0, 1.2),
        "elevation": (900, 500), "rainfall": (700, 250),
        "river_proximity": (15, 12), "pop_density": (400, 250),
        "temp_proxy": (28, 4),
    },
    "Balochistan": {
        "lat": (28.5, 2.0), "lon": (65.5, 2.2),
        "elevation": (900, 400), "rainfall": (150, 100),
        "river_proximity": (55, 35), "pop_density": (60, 45),
        "temp_proxy": (36, 4),
    },
    "Gilgit-Baltistan": {
        "lat": (35.7, 0.7), "lon": (74.8, 1.0),
        "elevation": (3800, 1600), "rainfall": (250, 120),
        "river_proximity": (10, 8), "pop_density": (30, 20),
        "temp_proxy": (18, 5),
    },
    "Azad Jammu & Kashmir": {
        "lat": (34.3, 0.6), "lon": (73.8, 0.6),
        "elevation": (1300, 600), "rainfall": (1300, 300),
        "river_proximity": (8, 6), "pop_density": (350, 200),
        "temp_proxy": (24, 4),
    },
    "Islamabad Capital Territory": {
        "lat": (33.7, 0.05), "lon": (73.1, 0.05),
        "elevation": (540, 20), "rainfall": (990, 40),
        "river_proximity": (12, 3), "pop_density": (2800, 100),
        "temp_proxy": (30, 2),
    },
}

SOIL_TYPES = ["alluvial", "clay", "sandy", "rocky"]


def _build_district_roster(rng: np.random.Generator) -> pd.DataFrame:
    """Assign 150 district rows across provinces proportional to their
    real district-count pools, cycling names with numeric suffixes if a
    province needs more rows than it has distinct real names for.
    """
    # Proportional allocation (roughly matches real relative district counts).
    weights = {
        "Punjab": 0.34, "Sindh": 0.20, "Khyber Pakhtunkhwa": 0.20,
        "Balochistan": 0.16, "Gilgit-Baltistan": 0.05,
        "Azad Jammu & Kashmir": 0.04, "Islamabad Capital Territory": 0.01,
    }
    provinces = list(weights.keys())
    probs = np.array([weights[p] for p in provinces])
    probs = probs / probs.sum()

    counts = rng.multinomial(N_DISTRICTS, probs)
    # Guarantee ICT gets exactly its 1 (capital territory, single district).
    ict_idx = provinces.index("Islamabad Capital Territory")
    if counts[ict_idx] == 0:
        counts[ict_idx] = 1
        counts[np.argmax(counts)] -= 1

    rows: list[dict[str, str]] = []
    for province, n in zip(provinces, counts):
        pool = PROVINCE_DISTRICTS[province]
        for i in range(n):
            if i < len(pool):
                name = pool[i]
            else:
                name = f"{pool[i % len(pool)]} Rural-{i // len(pool) + 1}"
            rows.append({"district_name": name, "province": province})

    roster = pd.DataFrame(rows)
    # Trim/pad to exactly N_DISTRICTS in the rare rounding-overflow case.
    if len(roster) > N_DISTRICTS:
        roster = roster.iloc[:N_DISTRICTS].reset_index(drop=True)
    elif len(roster) < N_DISTRICTS:
        deficit = N_DISTRICTS - len(roster)
        extra_province = rng.choice(provinces, size=deficit)
        extra_rows = [
            {
                "district_name": f"{PROVINCE_DISTRICTS[p][0]} Extension-{j}",
                "province": p,
            }
            for j, p in enumerate(extra_province)
        ]
        roster = pd.concat([roster, pd.DataFrame(extra_rows)], ignore_index=True)
    return roster


def _sample_province_conditioned_features(
    roster: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    """Sample geophysical features from per-province Gaussian profiles,
    clipped to realistic and/or Pakistan's real bounding-box ranges.
    """
    df = roster.copy()
    lat = np.empty(len(df))
    lon = np.empty(len(df))
    elevation = np.empty(len(df))
    rainfall = np.empty(len(df))
    river_prox = np.empty(len(df))
    pop_density = np.empty(len(df))
    temp_proxy = np.empty(len(df))

    for province, profile in PROVINCE_PROFILE.items():
        mask = (df["province"] == province).to_numpy()
        n = mask.sum()
        if n == 0:
            continue
        lat[mask] = rng.normal(*profile["lat"], size=n)
        lon[mask] = rng.normal(*profile["lon"], size=n)
        elevation[mask] = rng.normal(*profile["elevation"], size=n)
        rainfall[mask] = rng.normal(*profile["rainfall"], size=n)
        river_prox[mask] = rng.normal(*profile["river_proximity"], size=n)
        pop_density[mask] = rng.normal(*profile["pop_density"], size=n)
        temp_proxy[mask] = rng.normal(*profile["temp_proxy"], size=n)

    df["latitude"] = np.clip(lat, LAT_MIN, LAT_MAX).round(4)
    df["longitude"] = np.clip(lon, LON_MIN, LON_MAX).round(4)
    df["elevation_m"] = np.clip(elevation, 0, 8611).round(1)  # K2 = 8611m ceiling
    df["avg_annual_rainfall_mm"] = np.clip(rainfall, 30, None).round(1)
    df["river_proximity_km"] = np.clip(river_prox, 0.2, None).round(2)
    df["population_density"] = np.clip(pop_density, 3, None).round(1)
    df["_temp_proxy"] = temp_proxy  # internal only, used for heatwave label

    df["avg_summer_temp_celsius"] = np.clip(
        temp_proxy + rng.normal(0, 1.0, size=len(df)), 8, 52
    ).round(1)

    df["historical_flood_events"] = rng.poisson(
        lam=np.clip(6 - df["river_proximity_km"] / 15, 0.3, None)
    ).astype(int)
    df["historical_earthquake_events"] = rng.poisson(
        lam=np.clip((df["elevation_m"] / 2500) + (df["latitude"] - 28) / 6, 0.1, None)
    ).astype(int)
    df["historical_disasters"] = (
        df["historical_flood_events"] + df["historical_earthquake_events"]
    )

    df["soil_type"] = rng.choice(SOIL_TYPES, size=len(df), p=[0.35, 0.25, 0.25, 0.15])
    df["vegetation_index_ndvi"] = np.clip(
        rng.normal(0.45, 0.18, size=len(df))
        - (df["province"] == "Balochistan").to_numpy() * 0.12,
        0.02, 0.92,
    ).round(3)
    df["infrastructure_quality_score"] = np.clip(
        rng.normal(6.0, 1.8, size=len(df))
        - (df["population_density"] < 100).to_numpy() * 1.5,
        1, 10,
    ).round(1)

    return df


def _inject_extreme_outliers(
    df: pd.DataFrame, rng: np.random.Generator, frac: float = 0.04
) -> pd.DataFrame:
    """Inject a small fraction of physically extreme rows (climate-anomaly
    style outliers) so RobustScaler downstream has something to earn its
    keep against. Applied AFTER base sampling, BEFORE label derivation, so
    outliers correctly propagate into risk labels.
    """
    df = df.copy()
    n_outliers = max(2, int(len(df) * frac))
    idx = rng.choice(df.index, size=n_outliers, replace=False)

    for i in idx:
        kind = rng.choice(["mega_rain", "mega_heat", "mega_elevation"])
        if kind == "mega_rain":
            df.loc[i, "avg_annual_rainfall_mm"] *= rng.uniform(3.5, 5.5)
            df.loc[i, "river_proximity_km"] = rng.uniform(0.2, 2.0)
        elif kind == "mega_heat":
            df.loc[i, "avg_summer_temp_celsius"] = np.clip(
                df.loc[i, "avg_summer_temp_celsius"] + rng.uniform(8, 14), 8, 52
            )
        else:  # mega_elevation
            df.loc[i, "elevation_m"] = rng.uniform(5500, 8611)
            df.loc[i, "historical_earthquake_events"] += rng.integers(3, 7)

    logger.info("Injected %d extreme outlier rows (%.1f%%).", n_outliers, frac * 100)
    return df


def _derive_risk_labels(
    df: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    """Derive flood/heatwave/seismic risk labels from a weighted, noisy
    function of the physical features, then threshold UNEVENLY so 'High'
    risk is a genuine minority class (not a relabeled uniform split).
    """
    df = df.copy()

    def _minmax(s: pd.Series) -> pd.Series:
        return (s - s.min()) / (s.max() - s.min() + 1e-9)

    # --- Flood risk score ---
    flood_score = (
        0.35 * _minmax(df["avg_annual_rainfall_mm"])
        + 0.30 * (1 - _minmax(df["river_proximity_km"]))
        + 0.15 * _minmax(df["historical_flood_events"])
        + 0.10 * (1 - _minmax(df["elevation_m"]))
        + 0.10 * _minmax(df["population_density"])
        + rng.normal(0, 0.05, size=len(df))
    )

    # --- Heatwave risk score ---
    heatwave_score = (
        0.45 * _minmax(df["avg_summer_temp_celsius"])
        + 0.20 * (1 - _minmax(df["vegetation_index_ndvi"]))
        + 0.20 * (df["province"].isin(["Sindh", "Balochistan", "Punjab"])).astype(float)
        + 0.15 * (1 - _minmax(df["elevation_m"]))
        + rng.normal(0, 0.05, size=len(df))
    )

    # --- Seismic risk score ---
    seismic_score = (
        0.40 * _minmax(df["elevation_m"])
        + 0.25 * _minmax(df["historical_earthquake_events"])
        + 0.20 * (df["province"].isin(
            ["Gilgit-Baltistan", "Khyber Pakhtunkhwa", "Azad Jammu & Kashmir"]
        )).astype(float)
        + 0.15 * (1 - _minmax(df["infrastructure_quality_score"]))
        + rng.normal(0, 0.05, size=len(df))
    )

    def _threshold_minority_high(score: pd.Series, low_q: float, high_q: float) -> np.ndarray:
        """Threshold a continuous score into {0,1,2} using unequal
        quantile cuts so class 2 (High) ends up a clear minority (~12-18%).
        """
        lo_cut = score.quantile(low_q)
        hi_cut = score.quantile(high_q)
        labels = np.where(score >= hi_cut, 2, np.where(score >= lo_cut, 1, 0))
        return labels

    df["flood_risk"] = _threshold_minority_high(flood_score, 0.55, 0.85)
    df["heatwave_risk"] = _threshold_minority_high(heatwave_score, 0.55, 0.88)
    df["seismic_risk"] = _threshold_minority_high(seismic_score, 0.55, 0.87)

    df["overall_risk_score"] = np.clip(
        (
            3.3 * df["flood_risk"]
            + 3.3 * df["heatwave_risk"]
            + 3.4 * df["seismic_risk"]
        )
        + rng.normal(0, 0.3, size=len(df)),
        1, 10,
    ).round(1)

    return df.drop(columns=["_temp_proxy"])


def generate_dataset(seed: int = SEED, n_districts: int = N_DISTRICTS) -> pd.DataFrame:
    """Generate the full synthetic Pakistan multi-hazard districts dataset.

    Args:
        seed: RNG seed for full reproducibility.
        n_districts: Number of district rows to generate.

    Returns:
        A DataFrame with n_districts rows and all feature/label columns.
    """
    rng = np.random.default_rng(seed)
    logger.info("Generating %d synthetic district rows (seed=%d).", n_districts, seed)

    roster = _build_district_roster(rng)
    df = _sample_province_conditioned_features(roster, rng)
    df = _inject_extreme_outliers(df, rng)
    df = _derive_risk_labels(df, rng)

    df = df.drop_duplicates(subset=["district_name"], keep="first").reset_index(drop=True)
    if len(df) < n_districts:
        logger.warning(
            "Dropped %d duplicate district names; final row count = %d.",
            n_districts - len(df), len(df),
        )

    ordered_cols = [
        "district_name", "province", "latitude", "longitude", "elevation_m",
        "avg_annual_rainfall_mm", "river_proximity_km", "population_density",
        "avg_summer_temp_celsius", "historical_flood_events",
        "historical_earthquake_events", "historical_disasters", "soil_type",
        "vegetation_index_ndvi", "infrastructure_quality_score",
        "flood_risk", "heatwave_risk", "seismic_risk", "overall_risk_score",
    ]
    df = df[ordered_cols]

    for target in ["flood_risk", "heatwave_risk", "seismic_risk"]:
        dist = df[target].value_counts(normalize=True).sort_index().round(3)
        logger.info("%s class distribution (0=Low,1=Med,2=High): %s", target, dist.to_dict())

    return df


def main() -> None:
    df = generate_dataset()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    logger.info("Saved %d rows x %d columns -> %s", df.shape[0], df.shape[1], OUTPUT_PATH)
    logger.info("Provinces represented: %s", sorted(df["province"].unique().tolist()))


if __name__ == "__main__":
    main()
