"""Pakistan Multi-Hazard Risk Analyzer — Streamlit Dashboard.

Explainable-AI decision-support dashboard for flood, heatwave, and
seismic risk across 150 Pakistani districts. Loads the pre-trained
pipeline bundle produced by analysis.ipynb and exposes an interactive
scenario engine (slider-based feature simulation) plus SHAP-based
per-prediction explanations.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend, set BEFORE pyplot import.
# Required for server-side stability: Streamlit runs each session's script
# in its own thread, and matplotlib's default backend on some platforms
# is GUI-based and not thread-safe, causing intermittent native segfaults
# under concurrent Streamlit sessions. Agg is headless and thread-safe.

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st
from matplotlib import pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("ndma_dashboard")

DATA_PATH = Path("data/pakistan_districts.csv")
MODEL_PATH = Path("models/best_hazard_pipeline.pkl")

RISK_COLORS = {"Low": "#2E7D32", "Medium": "#F9A825", "High": "#C62828"}
RISK_ICONS = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}

TARGET_LABELS = {
    "flood_risk": "Flood Risk",
    "heatwave_risk": "Heatwave Risk",
    "seismic_risk": "Seismic Risk",
}

# Sliders: (min, max, step, help_text) per numeric feature exposed in the
# scenario engine. Not every numeric feature is user-tunable — identifying
# ones (lat/lon) and count-based history features stay fixed to the
# selected district's baseline to keep the scenario physically coherent.
SLIDER_CONFIG: dict[str, dict[str, Any]] = {
    "avg_annual_rainfall_mm": {"min": 30.0, "max": 3000.0, "step": 10.0, "label": "Annual Rainfall (mm)"},
    "elevation_m": {"min": 0.0, "max": 8611.0, "step": 10.0, "label": "Elevation (m)"},
    "avg_summer_temp_celsius": {"min": 8.0, "max": 52.0, "step": 0.5, "label": "Summer Temperature (°C)"},
    "river_proximity_km": {"min": 0.2, "max": 200.0, "step": 0.5, "label": "Distance to Nearest River (km)"},
    "population_density": {"min": 3.0, "max": 5000.0, "step": 10.0, "label": "Population Density (per km²)"},
    "vegetation_index_ndvi": {"min": 0.02, "max": 0.92, "step": 0.01, "label": "Vegetation Index (NDVI)"},
    "infrastructure_quality_score": {"min": 1.0, "max": 10.0, "step": 0.5, "label": "Infrastructure Quality (1-10)"},
}


# --------------------------------------------------------------------------
# Cached resource / data loaders
# --------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading trained hazard models...")
def load_model_bundle(path: str) -> dict[str, Any] | None:
    """Load the joblib model bundle once per server process.

    Returns:
        The deployment bundle dict, or None if the file is missing/corrupt.
    """
    try:
        bundle = joblib.load(path)
        logger.info("Loaded model bundle from %s (targets=%s)", path, bundle.get("targets"))
        return bundle
    except FileNotFoundError:
        logger.error("Model bundle not found at %s", path)
        return None
    except Exception:
        logger.exception("Failed to load model bundle at %s", path)
        return None


@st.cache_data(show_spinner="Loading district dataset...")
def load_district_data(path: str) -> pd.DataFrame | None:
    """Load the districts CSV once per server process (cached by content)."""
    try:
        df = pd.read_csv(path)
        logger.info("Loaded %d districts from %s", len(df), path)
        return df
    except FileNotFoundError:
        logger.error("District dataset not found at %s", path)
        return None
    except Exception:
        logger.exception("Failed to load district dataset at %s", path)
        return None


@st.cache_resource(show_spinner=False)
def build_shap_explainer(_pipeline, model_bundle_path: str, target: str):
    """Build (and cache) a SHAP TreeExplainer for one target's fitted model.

    Args:
        _pipeline: The fitted sklearn Pipeline (leading underscore tells
            Streamlit not to attempt to hash this unhashable object).
        model_bundle_path: Used only as a cache key so a changed model
            file invalidates this cached explainer.
        target: Hazard target name, also used as part of the cache key.
    """
    return shap.TreeExplainer(_pipeline.named_steps["clf"])


# --------------------------------------------------------------------------
# Prediction helpers
# --------------------------------------------------------------------------

def normalize_prediction(raw_pred: np.ndarray) -> int:
    """Normalize predict() output across model families.

    CatBoostClassifier.predict() returns shape (n, 1); RandomForest and
    XGBoost return flat shape (n,). Both are flattened to a plain int here
    so the rest of the app never has to special-case the winning family.
    """
    return int(np.asarray(raw_pred).ravel()[0])


def predict_all_hazards(
    bundle: dict[str, Any], feature_row: pd.DataFrame
) -> dict[str, dict[str, Any]]:
    """Run all three hazard pipelines on a single-row feature DataFrame.

    Returns:
        Mapping of target -> {"label": str, "class_idx": int, "probs": np.ndarray}.
    """
    results: dict[str, dict[str, Any]] = {}
    for target in bundle["targets"]:
        pipeline = bundle["pipelines"][target]
        raw_pred = pipeline.predict(feature_row)
        class_idx = normalize_prediction(raw_pred)
        probs = pipeline.predict_proba(feature_row)[0]
        results[target] = {
            "label": bundle["class_labels"][class_idx],
            "class_idx": class_idx,
            "probs": probs,
        }
    return results


def compute_shap_for_prediction(
    bundle: dict[str, Any], target: str, feature_row: pd.DataFrame, class_idx: int
):
    """Compute SHAP values for one prediction, returning a shap.Explanation
    object for the predicted class, already transformed through the
    pipeline's preprocessing step.

    The returned Explanation's `.data` holds RobustScaler-transformed
    values (correct for the SHAP math but unreadable to a non-technical
    viewer, e.g. elevation shown as `6.449` instead of `6449m`). Original
    human-readable values for numeric features are attached separately via
    `.display_data` so the waterfall plot can show real units.
    """
    pipeline = bundle["pipelines"][target]
    preprocessor = pipeline.named_steps["prep"]
    X_transformed = preprocessor.transform(feature_row)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()
    feature_names = preprocessor.get_feature_names_out()

    explainer = build_shap_explainer(pipeline, str(MODEL_PATH), target)
    shap_values = explainer.shap_values(X_transformed)

    # Normalize SHAP's multiclass output across library/version layouts:
    # either ndarray (n_samples, n_features, n_classes) or a list of
    # per-class (n_samples, n_features) arrays.
    if isinstance(shap_values, list):
        values_for_class = shap_values[class_idx][0]
        base_value = explainer.expected_value[class_idx]
    else:
        values_for_class = shap_values[0, :, class_idx]
        base_value = (
            explainer.expected_value[class_idx]
            if hasattr(explainer.expected_value, "__len__")
            else explainer.expected_value
        )

    # Build display values in original units: numeric features pull the
    # raw (pre-scaling) value from feature_row; one-hot categorical
    # columns display as their transformed 0/1 (already interpretable).
    display_values = np.array(X_transformed[0], dtype=object)
    for i, name in enumerate(feature_names):
        if name.startswith("num__"):
            raw_col = name.split("__", 1)[1]
            if raw_col in feature_row.columns:
                display_values[i] = round(float(feature_row[raw_col].iloc[0]), 2)

    explanation = shap.Explanation(
        values=values_for_class,
        base_values=base_value,
        data=X_transformed[0],
        feature_names=list(feature_names),
    )
    explanation.display_data = display_values
    return explanation


def clean_feature_name(raw_name: str) -> str:
    """Strip ColumnTransformer prefixes ('num__', 'cat__') and underscores
    for human-readable SHAP plot labels, with unit abbreviations kept
    lowercase/uppercase as conventionally written (mm, km, m, ndvi).
    """
    name = raw_name.split("__", 1)[-1]
    words = name.split("_")
    unit_overrides = {"mm": "mm", "km": "km", "m": "m", "ndvi": "NDVI", "celsius": "°C"}
    cleaned_words = [unit_overrides.get(w.lower(), w.capitalize()) for w in words]
    return " ".join(cleaned_words)


# --------------------------------------------------------------------------
# UI sections
# --------------------------------------------------------------------------

def render_sidebar(df: pd.DataFrame) -> pd.DataFrame:
    """Render the Scenario Engine sidebar and return the (possibly
    user-modified) single-row feature DataFrame to run predictions on.
    """
    st.sidebar.header("🎛️ Scenario Engine")
    st.sidebar.caption("Select a district, then simulate climate scenarios with the sliders below.")
    st.sidebar.divider()

    st.sidebar.markdown('<div class="ndma-sidebar-section">📍 District Selection</div>', unsafe_allow_html=True)
    district_names = sorted(df["district_name"].unique().tolist())
    default_idx = district_names.index("Karachi") if "Karachi" in district_names else 0

    selected_district = st.sidebar.selectbox(
        "Select District", district_names, index=default_idx, key="selected_district", label_visibility="collapsed"
    )

    if "last_district" not in st.session_state or st.session_state.last_district != selected_district:
        st.session_state.last_district = selected_district
        st.session_state.reset_sliders = True

    baseline_row = df[df["district_name"] == selected_district].iloc[0]

    st.sidebar.markdown(f"**Province:** {baseline_row['province']}")
    st.sidebar.markdown(f"**Soil Type:** {baseline_row['soil_type'].title()}")
    st.sidebar.divider()
    st.sidebar.markdown('<div class="ndma-sidebar-section">🌡️ Adjustable Parameters</div>', unsafe_allow_html=True)

    scenario_values: dict[str, float] = {}
    for feature, cfg in SLIDER_CONFIG.items():
        slider_key = f"slider_{feature}_{selected_district}"
        baseline_val = float(baseline_row[feature])
        scenario_values[feature] = st.sidebar.slider(
            cfg["label"],
            min_value=cfg["min"],
            max_value=cfg["max"],
            value=min(max(baseline_val, cfg["min"]), cfg["max"]),
            step=cfg["step"],
            key=slider_key,
            help=f"Baseline for {selected_district}: {baseline_val:.1f}",
        )

    st.sidebar.divider()
    if st.sidebar.button("↺ Reset to District Baseline", width='stretch'):
        for feature in SLIDER_CONFIG:
            st.session_state.pop(f"slider_{feature}_{selected_district}", None)
        st.rerun()

    # Build the full feature row: sliders override baseline for tunable
    # features; everything else (lat/lon, categoricals, history counts)
    # stays fixed to the selected district so the scenario stays coherent.
    feature_row = baseline_row.copy()
    for feature, value in scenario_values.items():
        feature_row[feature] = value

    return pd.DataFrame([feature_row]), selected_district, baseline_row


def render_command_center_css() -> None:
    """Inject scoped CSS for the elevated metric-card treatment.

    Targets Streamlit's own keyed-container wrapper class
    (`st-key-<key>`, stable public styling hook since Streamlit 1.3x) so
    the border/shadow/radius apply to the REAL container that holds the
    title, gauge, and confidence caption together — not a plain <div>
    that only wraps the title while the gauge renders as a sibling
    element outside it.
    """
    st.markdown(
        """
        <style>
        div[class*="st-key-hazard_card_"] {
            border-radius: 14px;
            padding: 0.9rem 1.1rem 0.4rem 1.1rem;
            border: 1px solid rgba(148, 163, 184, 0.28);
            background: linear-gradient(180deg, rgba(30, 41, 59, 0.045) 0%, rgba(30, 41, 59, 0.01) 100%);
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.10), 0 1px 3px rgba(15, 23, 42, 0.08);
        }
        .ndma-card-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 0.15rem;
        }
        .ndma-card-title {
            font-size: 0.82rem;
            font-weight: 600;
            letter-spacing: 0.02em;
            opacity: 0.75;
            text-transform: uppercase;
        }
        .ndma-card-badge {
            font-size: 0.72rem;
            font-weight: 700;
            padding: 0.15rem 0.55rem;
            border-radius: 999px;
            color: white;
            white-space: nowrap;
        }
        .ndma-card-confidence {
            font-size: 0.78rem;
            opacity: 0.65;
            text-align: center;
            margin-top: -0.6rem;
            padding-bottom: 0.6rem;
        }
        .ndma-sidebar-section {
            font-size: 0.95rem;
            font-weight: 700;
            letter-spacing: 0.01em;
            margin-top: 0.3rem;
            margin-bottom: 0.2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_risk_gauge(target: str, label: str, risk_score_pct: float, confidence_pct: float):
    """Build a Plotly gauge indicator for one hazard.

    The needle position is the model's weighted RISK SEVERITY (0-100,
    derived from class probabilities — same measure driving the radar
    chart), NOT the model's prediction confidence. Confidence is shown
    as a separate caption underneath. Conflating the two on one
    red/yellow/green gauge would visually imply "high confidence = high
    danger," which is false and would misinform a non-technical viewer —
    exactly the audience this dashboard exists to inform accurately.
    """
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=risk_score_pct,
            number={"suffix": "%", "font": {"size": 30}},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickmode": "array",
                    "tickvals": [0, 25, 50, 75, 100],
                    "ticktext": ["0", "25", "50", "75", "100"],
                    "tickwidth": 1,
                    "tickcolor": "rgba(128,128,128,0.4)",
                },
                "bar": {"color": "#1e293b", "thickness": 0.28},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 33], "color": "rgba(46, 125, 50, 0.55)"},
                    {"range": [33, 66], "color": "rgba(249, 168, 37, 0.55)"},
                    {"range": [66, 100], "color": "rgba(198, 40, 40, 0.55)"},
                ],
                "threshold": {
                    "line": {"color": "#1e293b", "width": 3},
                    "thickness": 0.85,
                    "value": risk_score_pct,
                },
            },
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=210,
        margin=dict(l=25, r=25, t=20, b=5),
    )
    return fig


def render_metrics(predictions: dict[str, dict[str, Any]]) -> None:
    """Render the top-row 3 hazard cards as elevated CSS containers, each
    with a Plotly gauge showing weighted risk severity (0-100) and the
    model's class-confidence shown separately beneath the gauge.
    """
    render_command_center_css()
    cols = st.columns(3)
    for col, target in zip(cols, ["flood_risk", "heatwave_risk", "seismic_risk"]):
        pred = predictions[target]
        label = pred["label"]
        confidence_pct = pred["probs"][pred["class_idx"]] * 100
        # Weighted risk score on a 0-100 scale: same probability-weighted
        # logic as the radar chart (sum of class_idx * prob), rescaled
        # from the model's 0-2 class range to a 0-100 gauge range.
        weighted_score = sum(i * p for i, p in enumerate(pred["probs"]))  # 0..2
        risk_score_pct = (weighted_score / 2) * 100

        with col:
            with st.container(border=True, key=f"hazard_card_{target}"):
                st.markdown(
                    f"""
                    <div class="ndma-card-header">
                        <span class="ndma-card-title">{TARGET_LABELS[target]}</span>
                        <span class="ndma-card-badge" style="background-color:{RISK_COLORS[label]};">
                            {RISK_ICONS[label]} {label}
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                fig = render_risk_gauge(target, label, risk_score_pct, confidence_pct)
                st.plotly_chart(fig, width="stretch", key=f"gauge_{target}", config={"displayModeBar": False})
                st.markdown(
                    f'<div class="ndma-card-confidence">Model confidence: {confidence_pct:.1f}%</div>',
                    unsafe_allow_html=True,
                )


def render_map_and_radar(
    feature_row: pd.DataFrame, district_name: str, predictions: dict[str, dict[str, Any]]
) -> None:
    """Render the district map (column 1) and multi-hazard radar chart
    (column 2) side by side.
    """
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📍 District Location")
        map_df = pd.DataFrame(
            {"lat": [float(feature_row["latitude"].iloc[0])],
             "lon": [float(feature_row["longitude"].iloc[0])]}
        )
        st.map(map_df, zoom=6, size=400, color="#C62828")
        st.caption(f"{district_name} — {feature_row['province'].iloc[0]}")

    with col2:
        st.subheader("📊 Multi-Hazard Risk Profile")
        categories = [TARGET_LABELS[t] for t in ["flood_risk", "heatwave_risk", "seismic_risk"]]
        # Radar value = probability-weighted risk score (0-2 scale) so the
        # chart reflects model confidence, not just the arg-max class.
        values = []
        for target in ["flood_risk", "heatwave_risk", "seismic_risk"]:
            probs = predictions[target]["probs"]
            weighted_score = sum(i * p for i, p in enumerate(probs))  # 0..2
            values.append(weighted_score)

        fig = go.Figure()
        fig.add_trace(
            go.Scatterpolar(
                r=values + [values[0]],
                theta=categories + [categories[0]],
                fill="toself",
                name="Risk Score",
                line_color="#C62828",
                fillcolor="rgba(198, 40, 40, 0.25)",
            )
        )
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 2], tickvals=[0, 1, 2],
                                        ticktext=["Low", "Medium", "High"])),
            showlegend=False,
            margin=dict(l=40, r=40, t=20, b=20),
            height=380,
        )
        st.plotly_chart(fig, width='stretch')


def render_shap_explanation(
    bundle: dict[str, Any], feature_row: pd.DataFrame, predictions: dict[str, dict[str, Any]]
) -> None:
    """Render the Explainable AI panel: a SHAP waterfall plot for whichever
    hazard has the highest predicted class (ties broken by confidence).
    """
    st.subheader("🔍 Explainable AI — Why This Prediction?")

    highest_target = max(
        predictions,
        key=lambda t: (predictions[t]["class_idx"], predictions[t]["probs"][predictions[t]["class_idx"]]),
    )
    pred = predictions[highest_target]

    st.markdown(
        f"Highest predicted hazard: **{TARGET_LABELS[highest_target]} — {pred['label']}** "
        f"({pred['probs'][pred['class_idx']] * 100:.1f}% confidence). "
        f"The waterfall below shows which features pushed this prediction "
        f"toward **{pred['label']}**, starting from the model's average "
        f"prediction (base value) on the left."
    )

    try:
        explanation = compute_shap_for_prediction(bundle, highest_target, feature_row, pred["class_idx"])
        explanation.feature_names = [clean_feature_name(n) for n in explanation.feature_names]

        fig, ax = plt.subplots(figsize=(10, 6))
        plt.sca(ax)
        shap.plots.waterfall(explanation, max_display=10, show=False)
        st.pyplot(fig, width='stretch')
        plt.close(fig)

        top_feature_idx = int(np.argmax(np.abs(explanation.values)))
        top_feature = explanation.feature_names[top_feature_idx]
        direction = "increased" if explanation.values[top_feature_idx] > 0 else "decreased"
        st.info(
            f"**Key driver:** `{top_feature}` {direction} the predicted "
            f"{TARGET_LABELS[highest_target].lower()} the most for this scenario."
        )
    except Exception:
        logger.exception("SHAP explanation failed for target=%s", highest_target)
        st.warning(
            "SHAP explanation could not be rendered for this scenario. "
            "The prediction above is still valid; only the explainability "
            "visualization failed."
        )


# --------------------------------------------------------------------------
# Main app
# --------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="NDMA Risk AI", layout="wide", page_icon="🛡️")

    st.title("🛡️ NDMA Pakistan — Multi-Hazard Risk Analyzer")
    st.caption(
        "AI-powered flood, heatwave, and seismic risk assessment across "
        "150+ Pakistani districts | Synthetic data, portfolio demonstration project"
    )

    bundle = load_model_bundle(str(MODEL_PATH))
    df = load_district_data(str(DATA_PATH))

    if bundle is None or df is None:
        st.error(
            "⚠️ **Model or data files not found.**\n\n"
            "This dashboard requires the trained model bundle and district "
            "dataset produced by the backend pipeline. Please run, in order:\n\n"
            "1. `python generate_data.py` — generates `data/pakistan_districts.csv`\n"
            "2. `jupyter nbconvert --to notebook --execute analysis.ipynb` "
            "(or run all cells in Jupyter) — trains models and saves "
            "`models/best_hazard_pipeline.pkl`\n\n"
            "Then restart this Streamlit app."
        )
        st.stop()

    if "reset_sliders" not in st.session_state:
        st.session_state.reset_sliders = False

    feature_row, selected_district, baseline_row = render_sidebar(df)

    model_features = bundle["numeric_features"] + bundle["categorical_features"]
    missing = [f for f in model_features if f not in feature_row.columns]
    if missing:
        st.error(f"⚠️ Feature mismatch between dataset and trained model: missing {missing}")
        st.stop()

    try:
        predictions = predict_all_hazards(bundle, feature_row[model_features])
    except Exception:
        logger.exception("Prediction failed")
        st.error(
            "⚠️ Prediction failed for the current scenario. This can happen "
            "if a slider value falls outside the range the model was "
            "trained on. Try resetting to the district baseline."
        )
        st.stop()

    st.divider()
    render_metrics(predictions)
    st.divider()
    render_map_and_radar(feature_row, selected_district, predictions)
    st.divider()
    render_shap_explanation(bundle, feature_row[model_features], predictions)

    with st.expander("📋 Full District Data Table"):
        st.dataframe(
            df[
                ["district_name", "province", "flood_risk", "heatwave_risk",
                 "seismic_risk", "overall_risk_score"]
            ]
            .replace({"flood_risk": {0: "Low", 1: "Medium", 2: "High"},
                       "heatwave_risk": {0: "Low", 1: "Medium", 2: "High"},
                       "seismic_risk": {0: "Low", 1: "Medium", 2: "High"}})
            .sort_values("overall_risk_score", ascending=False),
            width='stretch',
            height=300,
        )

    st.divider()
    st.caption(
        "⚠️ Built on synthetic data for demonstration purposes. Model "
        f"CV macro-F1: Flood={bundle['cv_macro_f1']['flood_risk']:.2f}, "
        f"Heatwave={bundle['cv_macro_f1']['heatwave_risk']:.2f}, "
        f"Seismic={bundle['cv_macro_f1']['seismic_risk']:.2f}. "
        "Not for operational disaster response use."
    )


if __name__ == "__main__":
    main()
