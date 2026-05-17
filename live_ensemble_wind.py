import re
import tempfile
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
import xarray as xr


# =========================
# Google Drive file IDs
# Replace these with your actual IDs
# =========================
GEFS_FILE_ID = "1kaJoW7CLXIzwfB75eQh3Y6tkPW3z3xnA"
ECMWF_FILE_ID = "1zp-ycSNN3Shk4oGLfAE4eRTCUm29zMoP"


MPS_TO_KT = 1.94384449


st.set_page_config(
    page_title="Everest Ensemble Wind",
    layout="wide",
)


def gdrive_download_url(file_id: str) -> str:
    return f"https://drive.google.com/uc?export=download&id={file_id}"


@st.cache_data(ttl=600)
def download_gdrive_file(file_id: str) -> str:
    url = gdrive_download_url(file_id)
    r = requests.get(url, timeout=60)
    r.raise_for_status()

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".nc")
    tmp.write(r.content)
    tmp.close()
    return tmp.name


@st.cache_data(ttl=600)
def load_dataset(file_id: str) -> xr.Dataset:
    path = download_gdrive_file(file_id)

    # decode_timedelta=False is robust to older xarray/netCDF step metadata
    ds = xr.open_dataset(path, decode_timedelta=False).load()
    return ds


def get_valid_time(ds: xr.Dataset) -> pd.DatetimeIndex:
    """
    Return forecast valid times in Nepal time.

    Prefer fxx + init_time because it is robust across GEFS/ECMWF and avoids
    pandas issues when valid_time is scalar or awkwardly encoded.
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

    return pd.DatetimeIndex(np.asarray(t).ravel()) + pd.Timedelta(hours=5, minutes=45)


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
        p90 = wspd
        p25 = wspd
        p75 = wspd
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


def plot_wind(df_gefs: pd.DataFrame, df_ecmwf: pd.DataFrame, units: str):
    factor = MPS_TO_KT if units == "kt" else 1.0
    ylabel = "Wind speed [kt]" if units == "kt" else "Wind speed [m s$^{-1}$]"

    fig, ax = plt.subplots(figsize=(13, 6))

    for df, label in [(df_gefs, "GEFS"), (df_ecmwf, "ECMWF ENS")]:
        x = df["time_npt"]
        ax.plot(x, df["median"] * factor, label=f"{label} median")
        ax.fill_between(
            x,
            df["p10"] * factor,
            df["p90"] * factor,
            alpha=0.18,
            label=f"{label} 10–90%",
        )

    for thresh in [20, 25, 30]:
        ax.axhline(thresh * factor, linestyle="--", linewidth=0.8)

    ax.set_title("Everest summit wind forecast")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Time [Nepal time]")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)

    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.xticks(rotation=45)
    plt.tight_layout()

    return fig


def plot_exceedance(df_gefs: pd.DataFrame, df_ecmwf: pd.DataFrame, threshold: int):
    col = f"p_exceed_{threshold}"

    fig, ax = plt.subplots(figsize=(13, 4))

    ax.plot(df_gefs["time_npt"], df_gefs[col], label="GEFS")
    ax.plot(df_ecmwf["time_npt"], df_ecmwf[col], label="ECMWF ENS")

    ax.set_title(f"Probability of summit wind ≥ {threshold} m s$^{{-1}}$")
    ax.set_ylabel("Probability [%]")
    ax.set_xlabel("Time [Nepal time]")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.xticks(rotation=45)
    plt.tight_layout()

    return fig


def init_string(ds: xr.Dataset) -> str:
    init = ds.attrs.get("init_time", "unknown")
    return str(init)


st.title("Everest Ensemble Wind Forecast")

if "PASTE_" in GEFS_FILE_ID or "PASTE_" in ECMWF_FILE_ID:
    st.error("Please replace GEFS_FILE_ID and ECMWF_FILE_ID in the script.")
    st.stop()

with st.spinner("Loading forecast data..."):
    ds_gefs = load_dataset(GEFS_FILE_ID)
    ds_ecmwf = load_dataset(ECMWF_FILE_ID)

df_gefs = ensemble_summary(ds_gefs, "GEFS")
df_ecmwf = ensemble_summary(ds_ecmwf, "ECMWF ENS")

col1, col2, col3 = st.columns(3)
col1.metric("GEFS init", init_string(ds_gefs) + " UTC")
col2.metric("ECMWF init", init_string(ds_ecmwf) + " UTC")
col3.metric("Displayed time", "Nepal time")

units = st.radio(
    "Wind-speed units",
    ["m s⁻¹", "kt"],
    horizontal=True,
)

plot_units = "kt" if units == "kt" else "m/s"

st.pyplot(plot_wind(df_gefs, df_ecmwf, plot_units))

threshold = st.selectbox(
    "Exceedance threshold",
    [20, 25, 30],
    index=1,
    format_func=lambda x: f"{x} m s⁻¹",
)

st.pyplot(plot_exceedance(df_gefs, df_ecmwf, threshold))

with st.expander("Show forecast table"):
    table = pd.concat([df_gefs, df_ecmwf], ignore_index=True)
    st.dataframe(table, use_container_width=True)
