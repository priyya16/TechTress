"""Streamlit frontend. All numbers come from src/pipeline.py and sibling modules."""

from __future__ import annotations

from io import StringIO

import pandas as pd
import plotly.express as px
import streamlit as st

from data_processor import DataValidationError
from explanations import (
    explain_gap,
    explain_prr_row,
    explain_signal_highlights,
    recommend_next_actions,
)
from pipeline import friendly_error, load_reports, run_dossier_pipeline, run_signal_pipeline
from quality import cases_for_pair, counts_by_year
from submission_checker import CHECKLIST_TITLE, DOSSIER_FORMAT_HELP, REPRESENTATIVE_CHECKLIST
from utils import (
    APP_SUBTITLE,
    APP_TITLE,
    CTD_DEMO_NOTE,
    DEMO_LEAF_NOTE,
    DISCLAIMER,
    HOME_BLURB,
    PRR_HELP,
    SAMPLE_ADVERSE_EVENTS,
    SAMPLE_DOSSIER_OUTLINE,
    SAMPLE_SIGNAL_DEMO,
    SPOTLIGHT_CTD_CODES,
    STATUS_BELOW,
    STATUS_INSUFFICIENT,
    STATUS_SIGNAL,
    STATUS_UNDEFINED,
    SYNTHETIC_BANNER,
    TEAM_NAME,
    VALUE_PROP,
)

st.set_page_config(page_title=APP_TITLE, page_icon="💊", layout="wide")

CUSTOM_CSS = """
<style>
    .stApp { background-color: #0B1220; color: #E8EEF8; }
    [data-testid="stSidebar"] { background-color: #101827; }
    [data-testid="stMetric"] {
        background: #151C2C;
        border: 1px solid #243044;
        border-radius: 12px;
        padding: 12px 14px;
    }
    [data-testid="stMetricLabel"] { color: #9AA8C3; }
    [data-testid="stMetricValue"] { color: #F4F7FB; }
    .tt-card {
        background: #151C2C;
        border: 1px solid #243044;
        border-radius: 14px;
        padding: 1rem 1.1rem;
        min-height: 8.5rem;
    }
    div[data-testid="stDataFrame"] { border: 1px solid #243044; border-radius: 10px; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

CHART_LAYOUT = dict(paper_bgcolor="#0B1220", plot_bgcolor="#151C2C", font_color="#E8EEF8")
STATUS_COLORS = {
    STATUS_SIGNAL: "#E07A5F",
    STATUS_BELOW: "#6EA8FE",
    STATUS_INSUFFICIENT: "#9AA8C3",
    STATUS_UNDEFINED: "#E0A14A",
}


def _disclaimer() -> None:
    with st.expander("Prototype scope (not FDA / not official ICH)", expanded=False):
        st.warning(DISCLAIMER)


def _show_error(exc: Exception) -> None:
    st.error(friendly_error(exc))


def _style_status(frame: pd.DataFrame, column: str = "Signal Status"):
    if frame.empty or column not in frame.columns:
        return frame

    def _color(val: str) -> str:
        if val == STATUS_SIGNAL or val == "Missing":
            return "background-color: #3A1F24; color: #F6C7BB; font-weight: 600"
        if val == "Present":
            return "background-color: #163328; color: #B8E6C9; font-weight: 600"
        if val == STATUS_INSUFFICIENT:
            return "background-color: #2A3142; color: #C9D3E3"
        if val == STATUS_UNDEFINED:
            return "background-color: #3A2E16; color: #F0D39A"
        return ""

    try:
        return frame.style.map(_color, subset=[column])
    except Exception:  # noqa: BLE001
        return frame


def render_home() -> None:
    st.caption(f"IBM Bob AI Innovation Hackathon 2026 · Team {TEAM_NAME} · AI Track")
    st.title(TEAM_NAME)
    st.markdown(f"### {APP_SUBTITLE}")
    st.write(HOME_BLURB)
    st.info(VALUE_PROP)
    _disclaimer()
    st.caption(SYNTHETIC_BANNER + ". This app does not process the FDA FAERS 20M+ database.")

    left, right = st.columns(2)
    with left:
        st.subheader("🔬 Signal Detection")
        st.write("Adverse-event reports → PRR analysis → Safety signals")
        st.caption("Find potential adverse-event safety signals using PRR.")
    with right:
        st.subheader("📋 Submission Readiness")
        st.write("CTD dossier → ICH M4-based checklist → Readiness score → Gap report")
        st.caption(CHECKLIST_TITLE + " — not complete official eCTD validation.")

    st.markdown("#### 2-minute demo")
    st.write(
        "1. **Signal Detection** → **Load Synthetic Demo Dataset** → preview → "
        "**Run Signal Analysis** → Novalexa / Liver Injury from the live PRR table. "
        "2. **Submission Readiness** → **Load Synthetic Demo Dossier** → module scores and gaps."
    )


def render_signal_detection() -> None:
    st.title("🔬 Signal Detection")
    st.caption("Adverse-event reports → PRR analysis → Safety signals. Not a medical conclusion.")
    _disclaimer()
    st.caption(PRR_HELP)
    st.caption(SYNTHETIC_BANNER)

    def _load_signal_demo() -> None:
        st.session_state["sig_ready"] = True
        st.session_state["sig_source"] = "demo"
        st.session_state["sig_ran"] = True

    st.button(
        "Load Synthetic Demo Dataset",
        type="primary",
        on_click=_load_signal_demo,
        key="load_signal_demo",
    )

    data_choice = st.radio(
        "Data source",
        ["Synthetic Demo Data", "Sample / Synthetic Data", "Upload CSV"],
        horizontal=True,
        help="Bundled files are Synthetic Demo Data, not FDA FAERS.",
        key="sig_radio",
    )
    if data_choice == "Sample / Synthetic Data":
        st.session_state["sig_source"] = "sample"
        st.session_state["sig_ready"] = True
        st.session_state["sig_ran"] = True
    elif data_choice == "Upload CSV":
        st.session_state["sig_source"] = "upload"

    uploaded = None
    if data_choice == "Upload CSV":
        uploaded = st.file_uploader(
            "Upload adverse-event CSV",
            type=["csv"],
            help="Required: a drug column (drug_name / drug / product) and an event column (adverse_event / event / pt).",
        )
    elif st.session_state.get("sig_source") == "demo":
        st.success("Loaded bundled `signal_detection_demo.csv`. " + SYNTHETIC_BANNER)
    elif st.session_state.get("sig_source") == "sample":
        st.caption("Using bundled **Synthetic Demo Data** (`adverse_events.csv`). Not FDA FAERS.")

    col_a, col_b = st.columns(2)
    with col_a:
        min_a = st.slider(
            "Minimum reports for the pair (a)",
            min_value=1,
            max_value=10,
            value=3,
            help="Changing this value recalculates signal status in the backend.",
        )
    with col_b:
        min_prr = st.slider(
            "PRR screening threshold",
            min_value=1.0,
            max_value=5.0,
            value=2.0,
            step=0.1,
            help="Changing this value reclassifies pairs using the same 2x2 counts.",
        )

    ready = bool(st.session_state.get("sig_ready")) or data_choice == "Upload CSV"
    if not ready:
        st.info("Click **Load Synthetic Demo Dataset** to preview the bundled Novalexa file, then run analysis.")
        return

    try:
        if data_choice == "Upload CSV":
            if uploaded is None:
                st.info("Upload a CSV to preview rows, then click **Run Signal Analysis**.")
                return
            source = uploaded
            source_label = "Uploaded file (not FAERS unless you supplied FAERS extracts)"
        elif st.session_state.get("sig_source") == "sample" or data_choice == "Sample / Synthetic Data":
            source = SAMPLE_ADVERSE_EVENTS
            source_label = "Sample / Synthetic Data"
        else:
            source = SAMPLE_SIGNAL_DEMO
            source_label = "Synthetic Demo Data"
        frame = load_reports(source)
    except DataValidationError as exc:
        _show_error(exc)
        return
    except Exception as exc:  # noqa: BLE001
        _show_error(exc)
        return

    st.subheader("Dataset preview")
    st.dataframe(frame.head(15), use_container_width=True)
    st.caption(
        f"{len(frame)} rows loaded. Each row is one report. {SYNTHETIC_BANNER}."
    )

    def _run_signal() -> None:
        st.session_state["sig_ran"] = True

    ran = st.button(
        "Run Signal Analysis",
        type="primary",
        on_click=_run_signal,
        key="run_signal",
    )
    if ran:
        st.session_state["sig_ran"] = True
    if not st.session_state.get("sig_ran"):
        st.info("Preview ready. Click **Run Signal Analysis** to calculate PRR and flags.")
        return

    try:
        with st.spinner("Calculating PRR from the uploaded/demo rows…"):
            payload = run_signal_pipeline(frame, min_a=min_a, min_prr=min_prr)
    except DataValidationError as exc:
        _show_error(exc)
        return
    except Exception as exc:  # noqa: BLE001
        _show_error(exc)
        return

    st.success(f"Analysis complete · source: **{source_label}**")
    st.caption(
        "Each row is counted as one report. Repeated drug-event pairs are kept so PRR uses the true case counts."
    )
    st.caption(SYNTHETIC_BANNER + ". Results are computed live; nothing is hardcoded.")

    clean_report = payload["clean_report"]
    with st.expander("Cleaning report", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Original rows", clean_report["original_rows"])
        c2.metric("Invalid / missing required", clean_report["invalid_or_missing_required"])
        c3.metric("Duplicates removed", clean_report["duplicate_rows_removed"])
        c4.metric("Rows after cleaning", clean_report["rows_after_cleaning"])
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Drug labels mapped", clean_report.get("drug_labels_mapped", 0))
        m2.metric("Event labels mapped", clean_report.get("event_labels_mapped", 0))
        m3.metric("Unique drugs after map", clean_report.get("unique_drugs_after_map", 0))
        m4.metric("Unique events after map", clean_report.get("unique_events_after_map", 0))
        st.caption(
            "Synonym map is a small prototype list (for example heptraclear → HeptraClear, "
            "hepatic injury → Liver Injury). It is not MedDRA."
        )

    stats = payload["stats"]
    results = payload["results"]
    signals = results[results["Signal Status"] == STATUS_SIGNAL]
    st.subheader("Summary")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Total Reports", stats["total_reports"])
    m2.metric("Drugs", stats["n_drugs"])
    m3.metric("Adverse Events", stats["n_events"])
    m4.metric("Drug-Event Pairs", stats["n_combinations"])
    m5.metric("Potential Signals", payload["n_signals"])
    m6.metric("Insufficient Data", payload.get("n_insufficient", 0))

    if not signals.empty:
        top = signals.iloc[0]
        st.subheader("Lead signal (from live PRR)")
        st.markdown(
            f"**{top['Drug']} + {top['Adverse Event']}** · "
            f"PRR **{top['PRR']}** · cases **{top['a']}** · **{top['Signal Status']}**"
        )
        orig = str(top.get("Original Event", "") or "")
        if orig and orig != str(top["Adverse Event"]):
            st.caption(f"Original Event: **{orig}** · Canonical Event: **{top['Adverse Event']}**")
        st.markdown("#### Why was this flagged?")
        for line in explain_signal_highlights(top, min_a=min_a, min_prr=min_prr):
            st.write(f"- {line}")
        st.caption(explain_prr_row(top))
        st.caption("Rule-based explanation from the 2x2 table. No LLM generated the PRR.")

    clustering = payload["clustering"]
    st.subheader("🧠 ML-Based Report Clustering")
    st.write(
        "TF-IDF converts report narratives into numerical text features, and "
        "KMeans groups similar narratives into clusters without predefined labels."
    )
    st.caption("Local scikit-learn only. No Gemini, Ollama, API key, or external AI service.")
    if clustering.get("method") != "tfidf_kmeans":
        st.info("Not enough narrative text for reliable ML clustering.")
        st.caption(clustering.get("explanation", ""))
    else:
        n_narr = clustering.get("n_narrative_reports", 0)
        n_cl = clustering.get("n_clusters", 0)
        c1, c2 = st.columns(2)
        c1.metric("Narrative reports analyzed", n_narr)
        c2.metric("Number of clusters", n_cl)
        summaries = clustering.get("summaries")
        if summaries is not None and not getattr(summaries, "empty", True):
            for rec in summaries.to_dict(orient="records"):
                cluster_no = int(rec.get("cluster", 0)) + 1
                st.markdown(f"**Cluster {cluster_no}**")
                st.write(f"Top keywords: {rec.get('top_terms', '')}")
                st.write(f"Reports: {rec.get('reports', 0)}")
                examples = rec.get("example_narratives") or []
                if isinstance(examples, str):
                    examples = [examples] if examples else []
                for snippet in list(examples)[:3]:
                    st.caption(f"Example: {snippet}")
        st.caption("These clusters group similar narratives. They do not replace PRR and are not diagnoses.")
        st.markdown(
            """
Narratives  
↓  
TF-IDF  
↓  
KMeans Clustering  
↓  
Similar Report Groups  
↓  
PRR Signal Analysis
"""
        )
        st.caption("TechTress uses **machine learning + statistical analysis**. PRR itself is not AI.")

    with st.expander("Pattern detection details (LDA / similar reports)", expanded=False):
        st.caption(
            "Local ML used here: TF-IDF + KMeans, LDA topics, cosine nearest neighbors, "
            "Isolation Forest, plus ROR and chi-square on the same 2x2 table. "
            "No external AI API is called."
        )
        st.info(clustering["explanation"])
        st.caption(
            f"Data used: `{clustering['used_column']}` · Method: `{clustering['method']}`"
        )
        if clustering["method"] == "tfidf_kmeans":
            extra = st.columns(2)
            extra[0].metric("Number of clusters", clustering.get("n_clusters", 0))
            sil = clustering.get("silhouette")
            extra[1].metric("Silhouette (cosine)", "n/a" if sil is None else sil)
        st.dataframe(clustering["summaries"], use_container_width=True)
        lda_topics = clustering.get("lda_topics")
        if lda_topics is not None and not getattr(lda_topics, "empty", True):
            st.markdown("**LDA topic words** (unsupervised; not diagnoses)")
            st.dataframe(lda_topics, use_container_width=True)
        similar = clustering.get("similar_reports")
        if similar is not None and not getattr(similar, "empty", True):
            st.markdown("**Similar narratives** (TF-IDF cosine nearest neighbors)")
            st.dataframe(similar, use_container_width=True)

    st.subheader("PRR analysis")
    st.caption(
        "Ranking: Potential Safety Signal first, then highest PRR, then highest a. "
        "Sliders above re-run backend classification; they do not invent new counts."
    )
    with st.expander("2x2 table definition"):
        st.markdown(
            """
            |  | Event | Other events |
            |---|---|---|
            | **Drug** | a | b |
            | **Other drugs** | c | d |

            **PRR** = [a / (a + b)] / [c / (c + d)]
            """
        )

    f1, f2, f3 = st.columns(3)
    status_options = ["All"] + sorted(results["Signal Status"].dropna().unique().tolist())
    drug_options = ["All"] + sorted(results["Drug"].dropna().unique().tolist())
    with f1:
        status_filter = st.selectbox("Filter by signal status", status_options)
    with f2:
        drug_filter = st.selectbox("Filter by drug", drug_options)
    agreement_options = ["All"]
    if "Method Agreement" in results.columns:
        agreement_options += sorted(results["Method Agreement"].dropna().unique().tolist())
    with f3:
        agreement_filter = st.selectbox("Filter by method agreement", agreement_options)

    visible = results.copy()
    if status_filter != "All":
        visible = visible[visible["Signal Status"] == status_filter]
    if drug_filter != "All":
        visible = visible[visible["Drug"] == drug_filter]
    if agreement_filter != "All" and "Method Agreement" in visible.columns:
        visible = visible[visible["Method Agreement"] == agreement_filter]
    st.caption(
        f"Showing {len(visible)} of {len(results)} backend-calculated rows. "
        "Method Agreement compares PRR and ROR using the same numeric threshold."
    )
    st.dataframe(_style_status(visible), use_container_width=True)
    if "Original Event" in visible.columns:
        st.caption(
            "Calculations use the **canonical** event name. **Original Event** is the "
            "label from the file (for example Hepatic Injury → Liver Injury)."
        )

    emerging = payload.get("emerging")
    emerging_reason = payload.get("emerging_reason", "ok")
    st.subheader("Emerging signals (time trend)")
    if emerging_reason == "no_year":
        st.info(
            "Time-based emergence could not be evaluated because this file has no "
            "`report_year` column. PRR signal detection above still applies."
        )
    elif emerging_reason == "insufficient_years":
        st.info(
            "Time-based emergence could not be evaluated because fewer than two "
            "distinct report years are present. PRR signal detection above still applies."
        )
    elif emerging is None or getattr(emerging, "empty", True):
        st.caption(
            "No PRR-flagged pair also showed recent-year growth (1.5× and at least "
            "+2 cases vs earlier years). A strong PRR signal is still a PRR signal "
            "even when it is not labeled emerging."
        )
    else:
        st.success("Emerging Signal — recent-year counts grew and PRR already met the screening rule.")
        st.dataframe(emerging, use_container_width=True)
        for rec in emerging.to_dict(orient="records"):
            with st.expander(f"{rec['Drug']} — {rec['Adverse Event']} (emerging)"):
                st.write(rec.get("Explanation", ""))

    cleaned = payload["cleaned"]
    st.subheader("Case-level drill-down")
    st.caption("Select a drug-event pair to see the actual report rows behind the 2x2 counts.")
    pair_labels = (
        results["Drug"].astype(str) + " · " + results["Adverse Event"].astype(str)
    ).tolist()
    default_pair = 0
    if not signals.empty:
        sig_label = str(signals.iloc[0]["Drug"]) + " · " + str(signals.iloc[0]["Adverse Event"])
        if sig_label in pair_labels:
            default_pair = pair_labels.index(sig_label)
    chosen = st.selectbox("Pair to inspect", pair_labels, index=min(default_pair, max(len(pair_labels) - 1, 0)))
    chosen_drug, chosen_event = chosen.split(" · ", 1)
    case_rows = cases_for_pair(cleaned, chosen_drug, chosen_event)
    st.metric("Cases in this pair", len(case_rows))
    chosen_row = results[
        (results["Drug"] == chosen_drug) & (results["Adverse Event"] == chosen_event)
    ]
    if not chosen_row.empty and chosen_row.iloc[0]["Signal Status"] == STATUS_SIGNAL:
        in_emerging = False
        if emerging is not None and not getattr(emerging, "empty", True):
            in_emerging = (
                (emerging["Drug"] == chosen_drug) & (emerging["Adverse Event"] == chosen_event)
            ).any()
        if in_emerging:
            st.success("Emerging Signal for this pair.")
        else:
            st.info("Strong PRR Signal for this pair — the time-growth emerging rule was not met.")

    st.dataframe(case_rows, use_container_width=True)

    st.subheader("Yearly Reporting Trend")
    if "report_year" not in cleaned.columns:
        st.info("No report_year column in this file, so a year trend cannot be drawn.")
    else:
        year_all = counts_by_year(cleaned)
        year_pair = counts_by_year(cleaned, drug=chosen_drug, event=chosen_event)
        if year_pair.empty:
            st.info("No dated reports for the selected pair.")
        else:
            trend_bits = [f"{int(r.report_year)} → {int(r.reports)}" for r in year_pair.itertuples()]
            st.write(" · ".join(trend_bits))
        t1, t2 = st.columns(2)
        with t1:
            if year_all.empty:
                st.info("report_year has no usable values.")
            else:
                fig_y = px.bar(
                    year_all,
                    x="report_year",
                    y="reports",
                    title="All reports by year",
                    color_discrete_sequence=["#6EA8FE"],
                )
                fig_y.update_layout(**CHART_LAYOUT)
                st.plotly_chart(fig_y, use_container_width=True)
        with t2:
            if not year_pair.empty:
                fig_p = px.bar(
                    year_pair,
                    x="report_year",
                    y="reports",
                    title=f"{chosen_drug} · {chosen_event} by year",
                    color_discrete_sequence=["#E07A5F"],
                )
                fig_p.update_layout(**CHART_LAYOUT)
                st.plotly_chart(fig_p, use_container_width=True)
        st.caption("Year charts are counts in this file only. They are not a full FAERS time-scan.")

    if "ML Anomaly" in results.columns:
        with st.expander("ML anomaly screening", expanded=False):
            st.caption(
                "Isolation Forest is trained on this file's pair features (counts, PRR, ROR, "
                "chi-square). Anomalous pair means unusual versus other pairs in this dataset, "
                "not a confirmed safety problem."
            )
            a1, a2 = st.columns(2)
            a1.metric("Isolation Forest anomalies", payload.get("n_anomalies", 0))
            if "Reviewer Priority Score" in results.columns:
                top_priority = results.sort_values("Reviewer Priority Score", ascending=False).head(1)
                if not top_priority.empty:
                    top = top_priority.iloc[0]
                    a2.metric(
                        "Highest priority pair",
                        f"{top['Drug']} · {top['Adverse Event']}",
                        f"score {top['Reviewer Priority Score']}",
                    )

    chart_source = signals if not signals.empty else results.dropna(subset=["PRR"]).head(10)
    if not chart_source.empty:
        chart_source = chart_source.copy()
        chart_source["pair"] = chart_source["Drug"] + " · " + chart_source["Adverse Event"]
        fig = px.bar(
            chart_source.head(10),
            x="PRR",
            y="pair",
            orientation="h",
            title="Top PRR pairs (from backend results)",
            color="Signal Status",
            color_discrete_map=STATUS_COLORS,
        )
        fig.update_layout(**CHART_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    freq = (
        payload["cleaned"]
        .groupby(["drug_name", "adverse_event"])
        .size()
        .reset_index(name="reports")
        .sort_values("reports", ascending=False)
        .head(12)
    )
    freq["pair"] = freq["drug_name"] + " · " + freq["adverse_event"]
    fig_f = px.bar(
        freq,
        x="reports",
        y="pair",
        orientation="h",
        title="Drug-event frequency",
        color_discrete_sequence=["#7BD389"],
    )
    fig_f.update_layout(**CHART_LAYOUT)
    st.plotly_chart(fig_f, use_container_width=True)

    st.subheader("Other flagged pairs")
    st.caption("These paragraphs are generated from the calculated 2x2 counts and PRR.")
    if signals.empty:
        st.write("No pairs met the current PRR and case-count thresholds.")
    else:
        for rec in signals.head(5).to_dict(orient="records"):
            with st.expander(f"{rec['Drug']} — {rec['Adverse Event']} (PRR {rec['PRR']})"):
                if rec.get("Original Event") and str(rec["Original Event"]) != str(rec["Adverse Event"]):
                    st.caption(
                        f"Original Event: {rec['Original Event']} · Canonical Event: {rec['Adverse Event']}"
                    )
                st.write(explain_prr_row(rec))

    st.download_button(
        "Download PRR results CSV",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name="prr_signal_results.csv",
        mime="text/csv",
        help="Downloads the full backend result table, not the filtered view.",
    )


def _module_label(full_name: str) -> str:
    return full_name.split("—")[0].strip()


def render_submission() -> None:
    st.title("📋 Submission Readiness")
    st.caption("CTD dossier → ICH M4-based checklist → Readiness score → Gap report")
    _disclaimer()
    st.info(CTD_DEMO_NOTE)
    st.caption(DEMO_LEAF_NOTE)

    with st.expander("Demo checklist (code · document · priority)", expanded=False):
        for module, spec in REPRESENTATIVE_CHECKLIST.items():
            st.markdown(f"**{module}** (module weight {int(spec['weight'] * 100)}%)")
            st.write(
                ", ".join(
                    f"{code} {section}"
                    for code, section, _priority in spec["sections"]
                )
            )

    def _load_ctd_demo() -> None:
        st.session_state["ctd_ready"] = True
        st.session_state["ctd_source"] = "sample"

    load_demo = st.button(
        "Load Synthetic Demo Dossier",
        type="primary",
        on_click=_load_ctd_demo,
        key="load_ctd_demo",
    )

    source_choice = st.radio(
        "Outline source",
        ["Sample / Synthetic Data", "Upload or paste"],
        horizontal=True,
        key="ctd_radio",
    )
    uploaded = None
    paste = ""
    if source_choice == "Upload or paste":
        uploaded = st.file_uploader(
            "Upload dossier outline (.txt or .csv)",
            type=["txt", "csv"],
            help=DOSSIER_FORMAT_HELP,
        )
        paste = st.text_area(
            "Or paste dossier sections (one per line)",
            height=180,
            placeholder="Module 3: Drug Substance\n5.9,Efficacy Summary,Missing,Demo item\n5.10,Integrated Benefit-Risk Summary,Missing,Demo item",
        )
    elif st.session_state.get("ctd_ready"):
        st.success("Loaded bundled synthetic dossier outline. Not a real regulatory dossier.")
    else:
        st.info("Click **Load Synthetic Demo Dossier** to score the bundled outline.")
        return

    try:
        if source_choice == "Sample / Synthetic Data":
            if not st.session_state.get("ctd_ready") and not load_demo:
                st.info("Click **Load Synthetic Demo Dossier** to begin.")
                return
            text = SAMPLE_DOSSIER_OUTLINE.read_text(encoding="utf-8")
            source_label = "Sample / Synthetic Data"
        elif uploaded is not None:
            raw_bytes = uploaded.getvalue()
            try:
                text = raw_bytes.decode("utf-8-sig")
            except UnicodeDecodeError:
                text = raw_bytes.decode("latin-1")
            source_label = "Uploaded file"
        elif paste.strip():
            text = paste
            source_label = "Pasted text"
        else:
            st.info("Upload, paste, or load the synthetic demo dossier.")
            return
        with st.spinner("Scoring the outline against the demo ICH M4 checklist…"):
            result = run_dossier_pipeline(text)
    except Exception as exc:  # noqa: BLE001
        _show_error(exc)
        return

    st.success(
        f"Outline source: **{source_label}** · parsed {result['provided_count']} section line(s)"
    )
    st.caption("Synthetic Demo Data — not a real regulatory dossier. Scores are computed live.")

    st.subheader("Overall Readiness")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Overall Readiness %", f"{result['overall_score']}")
    s2.metric("Present sections", f"{result['present_total']} / {result['expected_total']}")
    s3.metric("Missing Documents", result["missing_total"])
    s4.metric("High-Priority Gaps", result["missing_high"])
    st.progress(min(max(result["overall_score"] / 100.0, 0.0), 1.0))
    st.caption(result["score_formula"])

    st.subheader("Module scores")
    module_cols = st.columns(5)
    for idx, rec in enumerate(result["module_scores"].to_dict(orient="records")):
        with module_cols[idx]:
            st.metric(_module_label(rec["Module"]), f"{rec['Completeness %']}%")
            st.progress(min(max(rec["Completeness %"] / 100.0, 0.0), 1.0))

    if result["present_total"] == 0:
        st.info(
            "No Present documents matched the demo checklist (all expected items are Missing, "
            "or every mapped row was marked Missing). This 0% score is from the checklist, "
            "not a silent parse failure."
        )

    st.dataframe(result["module_scores"], use_container_width=True)
    fig = px.bar(
        result["module_scores"],
        x="Completeness %",
        y="Module",
        orientation="h",
        title="Module-wise readiness",
        color="Completeness %",
        color_continuous_scale="Tealgrn",
        range_color=[0, 100],
        range_x=[0, 100],
        text="Completeness %",
    )
    fig.update_traces(texttemplate="%{text:.0f}%", textposition="outside")
    fig.update_layout(**CHART_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

    gap = result["gap_report"]
    spotlight = gap[gap["Section Code"].astype(str).isin(SPOTLIGHT_CTD_CODES)].copy()
    st.subheader("Demo spotlight (live status)")
    st.caption("These rows are filtered from the computed gap report, not hardcoded.")
    if not spotlight.empty:
        show = spotlight[["Section Code", "Document", "Status", "Priority", "Notes"]]
        st.dataframe(_style_status(show, "Status"), use_container_width=True)

    st.subheader("Missing Documents / Gaps")
    missing = gap[gap["Status"] == "Missing"]
    st.dataframe(_style_status(missing, "Status"), use_container_width=True)

    st.subheader("Full gap report")
    st.dataframe(_style_status(gap, "Status"), use_container_width=True)

    for priority, count_key in (
        ("High", "missing_high"),
        ("Medium", "missing_medium"),
        ("Low", "missing_low"),
    ):
        subset = gap[(gap["Status"] == "Missing") & (gap["Priority"] == priority)]
        st.subheader(f"{priority}-priority gaps ({result[count_key]})")
        if subset.empty:
            st.success(f"No {priority.lower()}-priority representative sections are missing.")
        else:
            st.dataframe(_style_status(subset, "Status"), use_container_width=True)
            if priority == "High":
                for rec in subset.to_dict(orient="records"):
                    with st.expander(f"{rec['Module']} · {rec['Section']}"):
                        st.write(explain_gap(rec["Module"], rec["Section"], rec["Priority"]))

    st.subheader("Recommended next actions")
    st.caption("Rule-based recommendations from the calculated scores, not an external AI API.")
    for action in recommend_next_actions(result["missing_high"], result["overall_score"]):
        st.write(f"- {action}")

    buffer = StringIO()
    gap.to_csv(buffer, index=False)
    st.download_button(
        "Download gap report CSV",
        data=buffer.getvalue().encode("utf-8"),
        file_name="ctd_gap_report.csv",
        mime="text/csv",
        help="Module | Section Code | Document | Status | Notes | Priority | Recommendation",
    )


def main() -> None:
    st.sidebar.title(TEAM_NAME)
    st.sidebar.caption("Hackathon prototype · " + SYNTHETIC_BANNER)
    page = st.sidebar.selectbox(
        "Navigate",
        ["Home", "Signal Detection", "Submission Readiness"],
    )
    st.sidebar.markdown("---")
    st.sidebar.write("**🔬 Signal Detection** — Adverse-event reports → PRR → safety signals.")
    st.sidebar.write("**📋 Submission Readiness** — CTD outline → demo ICH M4 checklist → gaps.")
    st.sidebar.markdown("---")
    st.sidebar.caption(VALUE_PROP)
    st.sidebar.write(
        "Not FDA FAERS 20M+, not FDA certification, not complete ICH eCTD validation."
    )
    if page == "Home":
        render_home()
    elif page == "Signal Detection":
        render_signal_detection()
    else:
        render_submission()


if __name__ == "__main__":
    main()
