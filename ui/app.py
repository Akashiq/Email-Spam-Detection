"""Streamlit front-end for the Spam Detection API.

Run:  streamlit run ui/app.py        (API must be running on API_URL, default http://localhost:8000)
The UI never touches the model: everything goes through the FastAPI REST endpoints.
"""
import os

import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Spam Detector", page_icon="🛡️", layout="wide")

EXAMPLES = {
    "🎁 Prize scam": "WINNER!! You have been selected to receive a $1000 gift card. Click http://bit.ly/x9Z now!",
    "🏦 Phishing": "URGENT: your account is suspended. Verify your password at http://secure-login.info immediately",
    "💼 Work email": "Hi Sam, can we move the budget meeting to Friday? I will send the updated report this evening.",
    "🧪 Tricky ham": "Congratulations Alex, you won the team quiz! Prize is lunch on Friday.",
}

if "api_url" not in st.session_state:
    st.session_state.api_url = os.getenv("API_URL", "http://localhost:8000")
if "msg" not in st.session_state:
    st.session_state.msg = ""


# ------------------------------------------------------------------ API helpers
def call(method: str, path: str, **kwargs):
    """Call the FastAPI backend; show a friendly error and return None on failure."""
    url = st.session_state.api_url.rstrip("/") + path
    try:
        r = requests.request(method, url, timeout=30, **kwargs)
        r.raise_for_status()
        return r.json()
    except requests.HTTPError as exc:
        try:
            detail = exc.response.json().get("detail", "")
        except Exception:
            detail = exc.response.text[:200]
        st.error(f"API error {exc.response.status_code}: {detail}")
    except requests.RequestException:
        st.error(f"Cannot reach the API at {url}. Is FastAPI running?")
    return None


def quiet_health():
    try:
        return requests.get(st.session_state.api_url.rstrip("/") + "/health", timeout=3).json()
    except Exception:
        return None


def set_example(text: str):
    st.session_state.msg = text
    st.session_state.pop("result", None)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("🛡️ Spam Detector")
    st.text_input("API URL", key="api_url", help="Where the FastAPI service is running")
    health = quiet_health()
    if health is None:
        st.error("API unreachable")
    elif not health["model_loaded"]:
        st.warning("API up, but no model loaded.\nRun `python -m src.train`.")
    else:
        st.success("API online")
        st.caption(f"Model version: `{health['model_version']}`")
    st.divider()
    st.caption("UI → FastAPI → scikit-learn model\nEvery prediction is logged for monitoring.")

tab_single, tab_batch, tab_monitor = st.tabs(["🔍 Classify", "📂 Batch", "📊 Monitoring"])

# ------------------------------------------------------------------ tab 1: single message
with tab_single:
    st.subheader("Check a message")
    cols = st.columns(len(EXAMPLES))
    for col, (name, text) in zip(cols, EXAMPLES.items()):
        col.button(name, on_click=set_example, args=(text,), use_container_width=True)

    msg = st.text_area("Email / SMS text", key="msg", height=160, placeholder="Paste a message here…")

    if st.button("Analyze", type="primary", disabled=not msg.strip()):
        res = call("POST", "/predict", json={"text": msg})
        if res:
            st.session_state.result = res
            st.session_state.feedback_sent = False

    res = st.session_state.get("result")
    if res:
        p = res["spam_probability"]
        if res["is_spam"]:
            st.error(f"🚨 **SPAM** – {p:.1%} probability")
        else:
            st.success(f"✅ **Looks legitimate** – {1 - p:.1%} confidence")
        st.progress(p, text=f"Spam probability: {p:.1%}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Verdict", res["label"].upper())
        c2.metric("Latency", f"{res['latency_ms']:.2f} ms")
        c3.metric("Model version", res["model_version"])

        st.markdown("**Was this prediction correct?** Your answer is stored and used to monitor model quality.")
        if st.session_state.get("feedback_sent"):
            st.info("Thanks, feedback recorded ✔")
        else:
            b1, b2, _ = st.columns([1, 1, 4])
            pred = int(res["is_spam"])
            if b1.button("👍 Correct"):
                call("POST", "/feedback", json={"prediction_id": res["prediction_id"], "true_label": pred})
                st.session_state.feedback_sent = True
                st.rerun()
            if b2.button("👎 Wrong"):
                call("POST", "/feedback", json={"prediction_id": res["prediction_id"], "true_label": 1 - pred})
                st.session_state.feedback_sent = True
                st.rerun()

# ------------------------------------------------------------------ tab 2: batch
with tab_batch:
    st.subheader("Classify many messages")
    up = st.file_uploader("Upload a CSV", type="csv")
    pasted = st.text_area("…or paste one message per line", height=120, key="batch_text")
    texts, true_labels = [], None
    if up is not None:
        df_in = pd.read_csv(up, encoding="latin-1")
        guess = next((i for i, c in enumerate(df_in.columns) if str(c).lower() in {"text", "message", "v2", "email_text"}), 0)
        col = st.selectbox("Which column holds the text?", df_in.columns, index=guess)
        texts = df_in[col].astype(str).tolist()
        for lc in ("label", "v1", "Category"):          # optional ground truth -> accuracy
            if lc in df_in.columns:
                true_labels = df_in[lc].astype(str).str.lower().str.strip().tolist()
                break
    elif pasted.strip():
        texts = [ln.strip() for ln in pasted.splitlines() if ln.strip()]

    if len(texts) > 5000:
        st.warning("Only the first 5000 rows will be processed.")
        texts = texts[:5000]
        true_labels = true_labels[:5000] if true_labels else None

    if st.button(f"Classify {len(texts)} messages", type="primary", disabled=not texts):
        preds, bar = [], st.progress(0.0, text="Working…")
        for i in range(0, len(texts), 100):                 # API accepts max 100 per call
            out = call("POST", "/predict/batch", json={"texts": texts[i:i + 100]})
            if out is None:
                break
            preds += out["predictions"]
            bar.progress(min((i + 100) / len(texts), 1.0), text=f"{len(preds)}/{len(texts)}")
        bar.empty()
        if len(preds) == len(texts):
            result = pd.DataFrame({"text": texts, "label": [p["label"] for p in preds],
                                   "spam_probability": [p["spam_probability"] for p in preds]})
            if true_labels:
                result["true_label"] = true_labels
            st.session_state.batch = result

    result = st.session_state.get("batch")
    if result is not None:
        n_spam = int((result["label"] == "spam").sum())
        m1, m2, m3 = st.columns(3)
        m1.metric("Messages", len(result))
        m2.metric("Spam", n_spam)
        m3.metric("Spam rate", f"{n_spam / len(result):.1%}")
        if "true_label" in result:
            acc = (result["label"] == result["true_label"]).mean()
            st.metric("Accuracy vs your labels", f"{acc:.1%}")
        st.dataframe(result, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download results (CSV)", result.to_csv(index=False).encode(),
                           "spam_predictions.csv", "text/csv")

# ------------------------------------------------------------------ tab 3: monitoring
with tab_monitor:
    st.subheader("Live monitoring")
    st.button("🔄 Refresh")
    summary = call("GET", "/monitoring/summary")
    if summary:
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Predictions", summary["total_predictions"])
        k2.metric("Spam rate", f"{summary['spam_rate']:.1%}")
        k3.metric("Avg latency", f"{summary['avg_latency_ms']:.2f} ms")
        k4.metric("Labelled", summary["labeled_predictions"])
        acc = summary["live_accuracy"]
        k5.metric("Live accuracy", f"{acc:.1%}" if acc is not None else "n/a")

    recent = call("GET", "/predictions/recent", params={"limit": 200})
    if recent:
        rdf = pd.DataFrame(recent)
        left, right = st.columns(2)
        with left:
            st.markdown("**Spam-probability distribution (last 200)**")
            counts, edges = np.histogram(rdf["probability"], bins=10, range=(0, 1))
            st.bar_chart(pd.DataFrame({"predictions": counts}, index=[f"{e:.1f}" for e in edges[:-1]]))
        with right:
            st.markdown("**Predicted label mix**")
            st.bar_chart(rdf["prediction"].map({0: "ham", 1: "spam"}).value_counts())
        st.markdown("**Recent predictions**")
        view = pd.DataFrame({
            "time": rdf["ts"].str[:19].str.replace("T", " "),
            "text": rdf["text"].str.slice(0, 90),
            "predicted": rdf["prediction"].map({0: "ham", 1: "spam"}),
            "spam_prob": rdf["probability"].round(3),
            "true_label": rdf["true_label"].map({0: "ham", 1: "spam"}),
        })
        st.dataframe(view, use_container_width=True, hide_index=True)
    else:
        st.info("No predictions logged yet – classify something first.")

    st.divider()
    st.subheader("Performance-degradation check")
    st.caption("Compares live traffic with the training baseline: F1 drop, PSI, unseen-word rate, spam-rate shift.")
    last = st.number_input("Only analyse the last N predictions (0 = all)", min_value=0, value=0, step=50)
    if st.button("Run degradation check", type="primary"):
        report = call("GET", "/monitoring/report", params={"last": int(last) or None})
        if report:
            alerts = report.get("alerts", [])
            hard = [a for a in alerts if not a.startswith(("INFO", "NO_DATA"))]
            if report.get("degraded"):
                st.error(f"⚠️ DEGRADED – {len(hard)} alert(s). Consider retraining.")
            elif report["n_predictions"] == 0:
                st.info("No data for the current model version yet.")
            else:
                st.success("✅ HEALTHY – no degradation detected")
            for a in alerts:
                (st.info if a.startswith(("INFO", "NO_DATA")) else st.warning)(a)

            m = report.get("metrics", {})
            if m:
                live = m.get("live", {})
                table = pd.DataFrame([
                    ("F1 score", m.get("baseline_f1"), live.get("f1")),
                    ("Spam rate", m.get("baseline_spam_rate"), m.get("live_spam_rate")),
                    ("Unseen-word (OOV) rate", m.get("baseline_oov_rate"), m.get("live_oov_rate")),
                    ("Mean text length", m.get("baseline_mean_text_length"), m.get("live_mean_text_length")),
                ], columns=["Metric", "Baseline (training)", "Live"])
                st.dataframe(table.round(3), use_container_width=True, hide_index=True)
                p1, p2, p3 = st.columns(3)
                p1.metric("PSI (alert > 0.2)", f"{m.get('psi_probability', 0):.3f}")
                p2.metric("Labelled rows used", m.get("n_labeled", 0))
                p3.metric("p95 latency", f"{m.get('p95_latency_ms', 0):.2f} ms")