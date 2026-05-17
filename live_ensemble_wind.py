import tempfile

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
import xarray as xr


# =========================
# Google Drive file IDs
# =========================
GEFS_FILE_ID = "1kaJoW7CLXIzwfB75eQh3Y6tkPW3z3xnA"
ECMWF_FILE_ID = "1zp-ycSNN3Shk4oGLfAE4eRTCUm29zMoP"

# ECMWF ENS VO2max summary CSV
VO2_FILE_ID = "1QhsjGgTySXD3ERFBjY8qMi35QkarOAEl"


# =========================
# Constants
# =========================
MPS_TO_KT = 1.94384449

LTM_VO2MAX = 16.397887344862063
LT_MIN_VO2MAX = 15.686907307347052
LT_MAX_VO2MAX = 16.990437986057785


st.set_page_config(
    page_title="Everest Ensemble Forecast",
    layout="wide",
)


# =========================
# Google Drive helpers
# =========================
def gdrive_download_url(file_id: str) -> str:
    return f"https://drive.google.com/uc?export=download&id={file_id}"


def gdrive_csv_url(file_id: str) -> str:
    return f"https://drive.google.com/uc?id={file_id}"


@st.cache_data(ttl=600)
def download_gdrive_file(file_id: str, suffix: str = ".nc") -> str:
    url = gdrive_download_url(file_id)

    r = requests.get(url, timeout=60)
    r.raise_for_status()

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(r.content)
    tmp.close()

    return tmp.name


@st.cache_data(ttl=600)
def load_dataset(file_id: str) -> xr.Dataset:
    path = download_gdrive_file(file_id, suffix=".nc")
    return xr.open_dataset(path, decode_timedelta=False).load()


@st.cache_data(ttl=600)
def load_vo2_summary(file_id: str) -> pd.DataFrame:
    df = pd.read_csv(
        gdrive_csv_url(file_id),
        parse_dates=["time_utc", "time_npt"],
    )
    return df


# =========================
# Time helpers
# =========================
def get_valid_time(ds: xr.Dataset) -> pd.DatetimeIndex:
    """
    Return forecast valid times in Nepal time.
    """
    if "fxx" in ds.coords and "init_time" in ds.attrs:
        init = pd.to_datetime(ds.attrs["init_time"])
        fxx = np.asarray(ds["fxx"].values, dtype=float)
        t = init + pd.to_timedelta(fxx, unit="h")

    elif "valid_time" in ds:
        t = np.asarray(ds["valid_time"].values).squeeze()
        t = pd.to_datetime(t)

    else:
        raise ValueError("Could not determine forecast valid times.")

    return (
        pd.DatetimeIndex(np.asarray(t).ravel())
        + pd.Timedelta(hours=5, minutes=45)
    )


def init_string(ds: xr.Dataset) -> str:
    return str(ds.attrs.get("init_time", "unknown"))


# =========================
# Ensemble summary helper
# =========================
def ensemble_summary(ds: xr.Dataset, model_name: str) -> pd.DataFrame:

    wspd = ds["wspd_summit"]

    ens_dim = None
    for d in ["member", "number"]:
        if d in wspd.dims:
            ens_dim = d
            break

    if ens_dim is None:
        med = wspd
        p10 = wspd
        p25 = wspd
        p75 = wspd
        p90 = wspd

        p_exceed_20 = xr.zeros_like(wspd)
        p_exceed_25 = xr.zeros_like(wspd)
        p_exceed_30 = xr.zeros_like(wspd)

    else:
        med = wspd.median(ens_dim)

        p10 = wspd.quantile(0.10, ens_dim)
        p25 = wspd.quantile(0.25, ens_dim)
        p75 = wspd.quantile(0.75, ens_dim)
        p90 = wspd.quantile(0.90, ens_dim)

        p_exceed_20 = (wspd >= 20.0).mean(ens_dim) * 100
        p_exceed_25 = (wspd >= 25.0).mean(ens_dim) * 100
        p_exceed_30 = (wspd >= 30.0).mean(ens_dim) * 100

    return pd.DataFrame(
        {
            "time_npt": get_valid_time(ds),
            "median": np.asarray(med).squeeze(),
            "p10": np.asarray(p10).squeeze(),
            "p25": np.asarray(p25).squeeze(),
            "p75": np.asarray(p75).squeeze(),
            "p90": np.asarray(p90).squeeze(),
            "p_exceed_20": np.asarray(p_exceed_20).squeeze(),
            "p_exceed_25": np.asarray(p_exceed_25).squeeze(),
            "p_exceed_30": np.asarray(p_exceed_30).squeeze(),
            "model": model_name,
        }
    )


# =========================
# Plotting functions
# =========================
def plot_wind(
    df_gefs: pd.DataFrame,
    df_ecmwf: pd.DataFrame,
    units: str,
):

    factor = MPS_TO_KT if units == "kt" else 1.0

    ylabel = (
        "Wind speed [kt]"
        if units == "kt"
        else "Wind speed [m s$^{-1}$]"
    )

    fig, ax = plt.subplots(figsize=(13, 6))

    for df, label in [
        (df_gefs, "GEFS"),
        (df_ecmwf, "ECMWF ENS"),
    ]:

        x = df["time_npt"]

        ax.plot(
            x,
            df["median"] * factor,
            label=f"{label} median",
        )

        ax.fill_between(
            x,
            df["p10"] * factor,
            df["p90"] * factor,
            alpha=0.18,
            label=f"{label} 10–90%",
        )

    for thresh in [20, 25, 30]:
        ax.axhline(
            thresh * factor,
            linestyle="--",
            linewidth=0.8,
        )

    ax.set_title("Everest summit wind forecast")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Time [Nepal time]")

    ax.grid(True, alpha=0.3)

    ax.legend(ncol=2)

    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(
        mdates.DateFormatter("%b %d")
    )

    plt.xticks(rotation=45)
    plt.tight_layout()

    return fig


def plot_exceedance(
    df_gefs: pd.DataFrame,
    df_ecmwf: pd.DataFrame,
    threshold: int,
):

    col = f"p_exceed_{threshold}"

    fig, ax = plt.subplots(figsize=(13, 4))

    ax.plot(
        df_gefs["time_npt"],
        df_gefs[col],
        label="GEFS",
    )

    ax.plot(
        df_ecmwf["time_npt"],
        df_ecmwf[col],
        label="ECMWF ENS",
    )

    ax.set_title(
        f"Probability of summit wind ≥ {threshold} m s$^{{-1}}$"
    )

    ax.set_ylabel("Probability [%]")
    ax.set_xlabel("Time [Nepal time]")

    ax.set_ylim(0, 100)

    ax.grid(True, alpha=0.3)

    ax.legend()

    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(
        mdates.DateFormatter("%b %d")
    )

    plt.xticks(rotation=45)
    plt.tight_layout()

    return fig


def plot_vo2_ensemble(df: pd.DataFrame):

    fig, ax = plt.subplots(figsize=(13, 5))

    med = (
        df["vo2max_ml_kg_min_median"] / LTM_VO2MAX - 1.0
    ) * 100.0

    p10 = (
        df["vo2max_ml_kg_min_p10"] / LTM_VO2MAX - 1.0
    ) * 100.0

    p90 = (
        df["vo2max_ml_kg_min_p90"] / LTM_VO2MAX - 1.0
    ) * 100.0

    lt_min = (
        LT_MIN_VO2MAX / LTM_VO2MAX - 1.0
    ) * 100.0

    lt_max = (
        LT_MAX_VO2MAX / LTM_VO2MAX - 1.0
    ) * 100.0

    ax.plot(
        df["time_npt"],
        med,
        label="ECMWF ENS median",
    )

    ax.fill_between(
        df["time_npt"],
        p10,
        p90,
        alpha=0.2,
        label="ECMWF ENS 10–90%",
    )

    ax.axhline(
        0.0,
        linestyle="-",
        linewidth=0.8,
    )

    ax.axhline(
        lt_min,
        linestyle="--",
        linewidth=0.8,
        label="Historical no-O₂ summit range",
    )

    ax.axhline(
        lt_max,
        linestyle="--",
        linewidth=0.8,
    )

    ax.set_title("Everest summit VO₂max deviation")

    ax.set_ylabel(
        "ΔVO₂max [% vs LTM no-O₂ summit mean]"
    )

    ax.set_xlabel("Time [Nepal time]")

    ax.grid(True, alpha=0.3)

    ax.legend()

    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(
        mdates.DateFormatter("%b %d")
    )

    plt.xticks(rotation=45)
    plt.tight_layout()

    return fig


def plot_vo2_met(df: pd.DataFrame):

    fig, ax = plt.subplots(figsize=(13, 5))

    ax.plot(
        df["time_npt"],
        df["summit_t_C_median"],
        label="Temperature median",
    )

    ax.fill_between(
        df["time_npt"],
        df["summit_t_C_p10"],
        df["summit_t_C_p90"],
        alpha=0.15,
        label="Temperature 10–90%",
    )

    ax.set_ylabel("Temperature [°C]")

    ax.set_title(
        "ECMWF ENS summit temperature used for VO₂max forecast"
    )

    ax.set_xlabel("Time [Nepal time]")

    ax.grid(True, alpha=0.3)

    ax.legend()

    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(
        mdates.DateFormatter("%b %d")
    )

    plt.xticks(rotation=45)
    plt.tight_layout()

    return fig


# =========================
# Streamlit app
# =========================
st.title("Everest Ensemble Forecast")

required_ids = [
    GEFS_FILE_ID,
    ECMWF_FILE_ID,
]

if any("PASTE_" in x for x in required_ids):
    st.error(
        "Please replace GEFS_FILE_ID and ECMWF_FILE_ID in the script."
    )
    st.stop()


# =========================
# Load wind datasets
# =========================
with st.spinner("Loading ensemble wind forecast data..."):

    ds_gefs = load_dataset(GEFS_FILE_ID)
    ds_ecmwf = load_dataset(ECMWF_FILE_ID)

df_gefs = ensemble_summary(ds_gefs, "GEFS")
df_ecmwf = ensemble_summary(ds_ecmwf, "ECMWF ENS")


# =========================
# Metadata
# =========================
col1, col2, col3 = st.columns(3)

col1.metric(
    "GEFS init",
    init_string(ds_gefs) + " UTC",
)

col2.metric(
    "ECMWF init",
    init_string(ds_ecmwf) + " UTC",
)

col3.metric(
    "Displayed time",
    "Nepal time",
)


# =========================
# Wind plots
# =========================
st.header("Summit wind")

units = st.radio(
    "Wind-speed units",
    ["m s⁻¹", "kt"],
    horizontal=True,
)

plot_units = "kt" if units == "kt" else "m/s"

st.pyplot(
    plot_wind(
        df_gefs,
        df_ecmwf,
        plot_units,
    )
)

threshold = st.selectbox(
    "Exceedance threshold",
    [20, 25, 30],
    index=1,
    format_func=lambda x: f"{x} m s⁻¹",
)

st.pyplot(
    plot_exceedance(
        df_gefs,
        df_ecmwf,
        threshold,
    )
)


# =========================
# VO2max plots
# =========================
if "PASTE_" not in VO2_FILE_ID:

    with st.spinner(
        "Loading ECMWF ENS VO₂max forecast data..."
    ):

        df_vo2 = load_vo2_summary(VO2_FILE_ID)

    st.header("ECMWF ENS VO₂max")

    st.pyplot(
        plot_vo2_ensemble(df_vo2)
    )

    with st.expander(
        "Show temperature used for VO₂max"
    ):

        st.pyplot(
            plot_vo2_met(df_vo2)
        )

    with st.expander(
        "Show VO₂max summary table"
    ):

        st.dataframe(
            df_vo2,
            use_container_width=True,
        )

else:

    st.info(
        "ECMWF ENS VO₂max panel will appear once VO2_FILE_ID is added."
    )


# =========================
# Wind forecast table
# =========================
with st.expander("Show wind forecast table"):

    table = pd.concat(
        [df_gefs, df_ecmwf],
        ignore_index=True,
    )

    st.dataframe(
        table,
        use_container_width=True,
    )
