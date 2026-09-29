
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import streamlit as st

DASHBOARD_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DASHBOARD_DIR.parent

sys.path.insert(0, str(DASHBOARD_DIR))

from v2_backend import V2PredictionBackend

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "experimental_v2"
    / "three_class"
)

CLASS_NAMES = (
    "BENIGN",
    "DOS_DDOS",
    "RECONNAISSANCE",
)

st.set_page_config(
    page_title="Network Attack Forecasting",
    page_icon="???",
    layout="wide",
)

@st.cache_resource
def load_backend():
    return V2PredictionBackend()

@st.cache_resource
def load_test_data():
    X = np.load(
        DATA_DIR / "X_test.npy",
        mmap_mode="r",
    )
    y = np.load(
        DATA_DIR / "y_test.npy",
        mmap_mode="r",
    )

    if X.ndim != 3 or X.shape[1:] != (5, 15):
        raise ValueError(
            "Unexpected test input dimensions."
        )

    if len(X) != len(y):
        raise ValueError(
            "Test inputs and labels are misaligned."
        )

    return X, y

st.title("Network Attack Forecasting")
st.caption(
    "SIH experimental LSTM prototype | "
    "Offline test-window replay"
)

st.info(
    "This dashboard replays saved test windows. "
    "It is not connected to live network traffic. "
    "Warning thresholds and state-position lead "
    "times are experimental."
)

try:
    backend = load_backend()
    X, y = load_test_data()
except Exception as exc:
    st.error(
        "Could not load the prediction backend "
        "or test data."
    )
    st.exception(exc)
    st.stop()

st.sidebar.header("Replay controls")

sample_index = st.sidebar.number_input(
    "Test window index",
    min_value=0,
    max_value=len(X) - 1,
    value=0,
    step=1,
)

threshold_percent = st.sidebar.slider(
    "Experimental DDoS threshold (%)",
    min_value=1,
    max_value=100,
    value=10,
    step=1,
)

threshold = threshold_percent / 100.0

sample = np.asarray(
    X[int(sample_index)],
    dtype=np.float64,
)

try:
    result = backend.predict(
        sample,
        warning_threshold=threshold,
    )
except Exception as exc:
    st.error("Prediction failed.")
    st.exception(exc)
    st.stop()

actual_id = int(y[int(sample_index)])

actual_name = (
    CLASS_NAMES[actual_id]
    if 0 <= actual_id < len(CLASS_NAMES)
    else "UNKNOWN"
)

st.subheader("Next-state prediction")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Predicted next-state class",
    result["predicted_class"],
)

col2.metric(
    "DDoS probability",
    f'{result["ddos_probability"]:.2%}',
)

col3.metric(
    "Actual next-state class",
    actual_name,
)

if result["experimental_warning"]:
    st.warning(
        "EXPERIMENTAL DDOS WARNING: "
        "Predicted DDoS probability exceeds "
        "the selected threshold."
    )
else:
    st.success(
        "No DDoS threshold warning "
        "for this selected window."
    )

st.caption(
    "A threshold warning is not proof of an "
    "attack or guaranteed advance detection."
)

st.divider()

left, right = st.columns(2)

with left:
    st.subheader("Class probabilities")

    probabilities = pd.DataFrame({
        "Class": list(
            result["class_probabilities"].keys()
        ),
        "Probability": list(
            result["class_probabilities"].values()
        ),
    }).set_index("Class")

    st.bar_chart(probabilities)

with right:
    st.subheader("Prediction details")

    st.write({
        "Test window": int(sample_index),
        "Warning threshold": (
            f"{threshold_percent}%"
        ),
        "Predicted class": (
            result["predicted_class"]
        ),
        "Actual target class": actual_name,
        "Experimental warning": (
            result["experimental_warning"]
        ),
    })

st.divider()

st.subheader("Five input network states")

feature_names = ['Avg Flow Packets/s', 'Std Flow Packets/s', 'Avg Flow Bytes/s', 'Std Flow Bytes/s', 'Avg Flow Duration', 'Avg Average Packet Size', 'Avg SYN Flag Count', 'Avg ACK Flag Count', 'Avg RST Flag Count', 'Avg FIN Flag Count', 'Unique Destination Port Count', 'Mean Safe Packet Ratio', 'Mean Safe Byte Ratio', 'Mean Flow IAT Std', 'Mean IAT Variability']

input_frame = pd.DataFrame(
    sample,
    columns=feature_names,
    index=[
        f"Input state {i+1}"
        for i in range(5)
    ],
)

st.dataframe(
    input_frame,
    width="stretch",
)

st.subheader("Forecasted next-state features")

forecast = np.asarray(
    result["predicted_next_state"],
    dtype=np.float64,
)

forecast_frame = pd.DataFrame({
    "Feature": feature_names,
    "Last input state": sample[-1],
    "Predicted next state": forecast,
}).set_index("Feature")

# Compare features on the training scaler's
# standardized scale, not their original units.
scaled_comparison = backend.scaler.transform(
    np.vstack([sample[-1], forecast])
)

scaled_frame = pd.DataFrame(
    scaled_comparison.T,
    index=feature_names,
    columns=[
        "Last input state",
        "Predicted next state",
    ],
)

st.caption(
    "Standardized feature values using the "
    "training-data scaler. These are not "
    "original measurement units."
)

st.bar_chart(
    scaled_frame,
    horizontal=True,
)

with st.expander(
    "View exact next-state feature values"
):
    st.dataframe(
        forecast_frame,
        width="stretch",
    )

st.divider()

st.subheader("Experimental Early-warning Demo")

st.info(
    "This is one previously examined test example, "
    "not an independent evaluation or live prediction."
)

demo_onset = 428
demo_threshold = 0.10
demo_rows = []

if len(X) <= demo_onset + 2:
    st.error("Expected demo test indices are unavailable.")
else:
    for offset in range(-5, 3):
        index = demo_onset + offset

        prediction = backend.predict(
            np.asarray(X[index]),
            warning_threshold=demo_threshold,
        )

        actual_id = int(y[index])

        demo_rows.append({
            "Position": offset,
            "Actual class": CLASS_NAMES[actual_id],
            "DDoS probability (%)": (
                prediction["ddos_probability"] * 100
            ),
            "Warning": prediction["experimental_warning"],
        })

    demo_frame = pd.DataFrame(demo_rows)

    st.line_chart(
        demo_frame.set_index("Position")[
            ["DDoS probability (%)"]
        ]
    )

    st.dataframe(
        demo_frame,
        width="stretch",
        hide_index=True,
        column_config={
            "DDoS probability (%)": st.column_config.NumberColumn(
                "DDoS probability (%)",
                format="%.2f%%",
            )
        },
    )

    precursor = demo_frame[
        (demo_frame["Position"] < 0)
        & (demo_frame["Warning"])
    ]

    if not precursor.empty:
        earliest = int(precursor["Position"].min())
        st.warning(
            f"Experimental warning observed at position "
            f"{earliest}, before the observed onset at 0."
        )

    st.caption(
        "Position 0 is the observed DDoS onset. "
        "Negative positions are earlier dataset states. "
        "This example was previously inspected; "
        "chronological timing and seconds-based "
        "advance warning are not established."
    )

st.divider()

st.subheader("Evaluation context")

st.markdown(
    """
    **Internal V2 test classification**
    - Accuracy: 99.7525%
    - Macro F1: 0.97473
    - DDoS recall: 285/287

    **Experimental early warning at 10%**
    - Validation: 2/3 observed onsets
      had a precursor warning.
    - Test: 1/1 observed onset
      had a precursor warning.
    - Unrelated warning episodes:
      46 validation and 114 test.

    These observations do not establish
    real-world early-warning reliability.
    """
)

st.caption(
    "The model predicts the next state in "
    "dataset row order. Chronological ordering "
    "and seconds-based lead time have not "
    "been verified."
)



# STEP 103: EXTERNAL EVALUATION SECTION

st.divider()
st.subheader("External CIC-IDS2018 Evaluation")

st.info(
    "Additional external generalization evaluation on a "
    "previously inspected dataset. This is not untouched "
    "independent validation."
)

import json
from pathlib import Path

external_metrics_path = (
    Path(__file__).resolve().parent.parent
    / "external_test"
    / "results"
    / "v2_external_evaluation"
    / "v2_external_metrics.json"
)

if external_metrics_path.is_file():

    external_metrics = json.loads(
        external_metrics_path.read_text(
            encoding="utf-8"
        )
    )

    ddos_metrics = external_metrics["classes"]["DOS_DDOS"]

    metric_col1, metric_col2, metric_col3 = st.columns(3)

    metric_col1.metric(
        "External accuracy",
        f'{external_metrics["accuracy"]:.2%}'
    )

    metric_col2.metric(
        "Balanced accuracy",
        f'{external_metrics["balanced_accuracy"]:.2%}'
    )

    metric_col3.metric(
        "DDoS recall",
        f'{ddos_metrics["recall"]:.2%}'
    )

    st.caption(
        "Overall accuracy is influenced by the large "
        "number of benign samples. DDoS recall is "
        "substantially lower."
    )

    st.markdown("**DDoS detection**")

    detection_col1, detection_col2 = st.columns(2)

    detection_col1.metric(
        "DDoS precision",
        f'{ddos_metrics["precision"]:.2%}'
    )

    detection_col2.metric(
        "DDoS warning rate at 10%",
        f'{external_metrics["ddos_warning_rate"]:.2%}'
    )

    st.metric(
        "Benign warning rate at 10%",
        f'{external_metrics["benign_warning_rate"]:.2%}'
    )

    st.markdown("**Next-state forecasting**")

    forecast_col1, forecast_col2 = st.columns(2)

    forecast_col1.metric(
        "V2 scaled MSE",
        f'{external_metrics["next_state_scaled_mse"]:.2f}'
    )

    forecast_col2.metric(
        "Persistence baseline scaled MSE",
        f'{external_metrics["persistence_scaled_mse"]:.2f}'
    )

    st.caption(
        "Both forecasting errors use standardized feature "
        "values. Lower MSE is better. External feature "
        "distribution shift remains a significant concern."
    )

    st.warning(
        "The external DDoS recall is low. This dashboard "
        "is an experimental offline replay prototype, "
        "not a validated live intrusion detection system."
    )

else:

    st.warning(
        "External evaluation metrics file not found."
    )
