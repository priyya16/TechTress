"""Streamlit frontend. All numbers come from src/pipeline.py and sibling modules."""

from __future__ import annotations

from io import StringIO

import pandas as pd
import plotly.express as px
import streamlit as st

from data_processor import DataValidationError
from quality import cases_for_pair, counts_by_year
from explanations import explain_gap, explain_prr_row, recommend_next_actions
from pipeline import friendly_error, load_reports, run_dossier_pipeline, run_signal_pipeline
from submission_checker import REPRESENTATIVE_CHECKLIST
from utils import (
    APP_TITLE,
    DISCLAIMER,
    PRR_HELP,
    SAMPLE_ADVERSE_EVENTS,
    SAMPLE_DOSSIER_OUTLINE,
    STATUS_SIGNAL,
    TEAM_NAME,
)

st.set_page_config(page_title=APP_TITLE, page_icon="💊", layout="wide")

CUSTOM_CSS = """
<style>
    .stApp { background-color: #0B1220; color: #E8EEF8; }
    [data-testid="stSidebar"] { background-color: #101827; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

CHART_LAYOUT = dict(paper_bgcolor="#0B1220", plot_bgcolor="#151C2C", font_color="#E8EEF8")


def _disclaimer() -> None:
    st.warning(DISCLAIMER)


def _show_error(exc: Exception) -> None:
    st.error(friendly_error(exc))


def render_home() -> None:
    st.caption(f"IBM Bob AI Innovation Hackathon 2026 · Team {TEAM_NAME} · AI Track")
    st.title("Drug Safety Signal Detector")
    st.markdown("### & Regulatory Submission Readiness Checker")
    _disclaimer()
    left, right = st.columns(2)
    with left:
        st.subheader("Mode 1 — Signal Detection")
        st.write(
            "Upload representative adverse-event reports or use sample / synthetic "
            "data. The backend validates the file, calculates PRR/ROR/chi-square, "
            "runs TF-IDF clustering and Isolation Forest ranking, and returns "
            "**Potential Safety Signals**."
        )
        st.caption(PRR_HELP)
    with right:
        st.subheader("Mode 2 — Submission Readiness")
        st.write(
            "Upload, paste, or load a representative CTD outline. The backend maps "
            "sections to five ICH M4 modules, scores completeness, and builds a "
            "downloadable gap report."
        )
        st.caption(
            "Representative prototype checklist — not an exhaustive regulatory submission checklist."
        )
    st.markdown("#### Demo path")
    st.write(
        "1. Open **Signal Detection** (sample data is the default) → review PRR, "
        "filters, charts → download CSV. "
        "2. Open **Submission Readiness** → review module scores and gaps → download the gap report."
    )


def render_signal_detection() -> None:
    st.title("Signal Detection")
    st.caption("Screen representative adverse-event data with PRR. Not a medical conclusion.")
    _disclaimer()
    st.info(PRR_HELP)

    data_choice = st.radio(
        "Data source",
        ["Sample / Synthetic Data", "Upload CSV"],
        horizontal=True,
        help="Sample data is synthetic and is not FDA FAERS.",
    )
    uploaded = None
    if data_choice == "Upload CSV":
        uploaded = st.file_uploader(
            "Upload adverse-event CSV",
            type=["csv"],
            help="Required columns: drug_name, adverse_event.",
        )
    else:
        st.caption("Using bundled Sample / Synthetic Data (not FDA FAERS).")

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

    try:
        use_sample = data_choice == "Sample / Synthetic Data"
        if uploaded is None and not use_sample:
            st.info("Upload a CSV or switch Data source to Sample / Synthetic Data.")
            return
        source = SAMPLE_ADVERSE_EVENTS if use_sample else uploaded
        frame = load_reports(source)
        source_label = "Sample / Synthetic Data" if use_sample else "Uploaded file"
        payload = run_signal_pipeline(frame, min_a=min_a, min_prr=min_prr)
    except DataValidationError as exc:
        _show_error(exc)
        return
    except Exception as exc:  # noqa: BLE001
        _show_error(exc)
        return

    st.success(f"Dataset source: **{source_label}**")
    if source_label.startswith("Sample"):
        st.warning(
            "This is **Sample / Synthetic Data**. It is not the complete FDA FAERS dataset."
        )

    st.subheader("Dataset preview")
    st.dataframe(frame.head(15), use_container_width=True)

    clean_report = payload["clean_report"]
    with st.expander("Cleaning report", expanded=True):
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
    st.subheader("Dataset statistics")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total reports", stats["total_reports"])
    m2.metric("Drugs", stats["n_drugs"])
    m3.metric("Adverse events", stats["n_events"])
    m4.metric("Drug-event combinations", stats["n_combinations"])
    m5.metric("Potential safety signals", payload["n_signals"])

    st.caption(
        "Local ML used here: TF-IDF + KMeans, LDA topics, cosine nearest neighbors, "
        "Isolation Forest, plus ROR and chi-square on the same 2x2 table. "
        "No external AI API is called."
    )

    clustering = payload["clustering"]
    st.subheader("Pattern detection")
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
    if clustering["method"] == "tfidf_kmeans":
        cluster_counts = (
            clustering["frame"]["cluster"].value_counts().sort_index().reset_index()
        )
        cluster_counts.columns = ["cluster", "reports"]
        fig_c = px.bar(
            cluster_counts,
            x="cluster",
            y="reports",
            title="Cluster distribution",
            color_discrete_sequence=["#6EA8FE"],
        )
        fig_c.update_layout(**CHART_LAYOUT)
        st.plotly_chart(fig_c, use_container_width=True)

    results = payload["results"]
    signals = results[results["Signal Status"] == STATUS_SIGNAL]
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
    st.dataframe(visible, use_container_width=True)

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
    st.dataframe(case_rows, use_container_width=True)

    st.subheader("Time trend")
    if "report_year" not in cleaned.columns:
        st.info("No report_year column in this file, so a year trend cannot be drawn.")
    else:
        year_all = counts_by_year(cleaned)
        year_pair = counts_by_year(cleaned, drug=chosen_drug, event=chosen_event)
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
            if year_pair.empty:
                st.info("No dated reports for the selected pair.")
            else:
                fig_p = px.bar(
                    year_pair,
                    x="report_year",
                    y="reports",
                    title=f"{chosen_drug} · {chosen_event} by year",
                    color_discrete_sequence=["#E07A5F"],
                )
                fig_p.update_layout(**CHART_LAYOUT)
                st.plotly_chart(fig_p, use_container_width=True)
        st.caption(
            "Year charts are counts in this file only. They are not a full FAERS time-scan."
        )

    if "ML Anomaly" in results.columns:
        st.subheader("ML anomaly screening")
        st.caption(
            "Isolation Forest is trained on this file's pair features (counts, PRR, ROR, "
            "chi-square). Anomalous pair means unusual versus other pairs in this dataset, "
            "not a confirmed safety problem. Reviewer Priority Score combines scaled PRR, "
            "chi-square, case count, and the anomaly flag."
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
        anomaly_counts = results["ML Anomaly"].value_counts().reset_index()
        anomaly_counts.columns = ["ML Anomaly", "count"]
        fig_a = px.bar(
            anomaly_counts,
            x="ML Anomaly",
            y="count",
            title="Isolation Forest labels",
            color_discrete_sequence=["#E07A5F"],
        )
        fig_a.update_layout(**CHART_LAYOUT)
        st.plotly_chart(fig_a, use_container_width=True)
        if "Reviewer Priority Score" in results.columns:
            rank_src = results.dropna(subset=["Reviewer Priority Score"]).copy()
            rank_src["pair"] = rank_src["Drug"] + " · " + rank_src["Adverse Event"]
            fig_p = px.bar(
                rank_src.sort_values("Reviewer Priority Score", ascending=False).head(10),
                x="Reviewer Priority Score",
                y="pair",
                orientation="h",
                title="Reviewer priority score (local ML ranking)",
                color="ML Anomaly",
            )
            fig_p.update_layout(**CHART_LAYOUT)
            st.plotly_chart(fig_p, use_container_width=True)

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
            color_discrete_map={
                STATUS_SIGNAL: "#E07A5F",
                "Below Threshold": "#6EA8FE",
                "Insufficient Data": "#9AA8C3",
            },
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

    status_counts = results["Signal Status"].value_counts().reset_index()
    status_counts.columns = ["Signal Status", "count"]
    fig_s = px.pie(
        status_counts,
        names="Signal Status",
        values="count",
        title="Signal status counts",
        color_discrete_sequence=["#E07A5F", "#6EA8FE", "#E0A14A", "#9AA8C3"],
    )
    fig_s.update_layout(paper_bgcolor="#0B1220", font_color="#E8EEF8")
    st.plotly_chart(fig_s, use_container_width=True)

    st.subheader("Rule-based reviewer explanations")
    st.caption(
        "These paragraphs are generated from the calculated 2x2 counts and PRR. "
        "They are not calls to an external AI model."
    )
    if signals.empty:
        st.write("No pairs met the current PRR and case-count thresholds.")
    else:
        for rec in signals.head(5).to_dict(orient="records"):
            with st.expander(f"{rec['Drug']} — {rec['Adverse Event']} (PRR {rec['PRR']})"):
                st.write(explain_prr_row(rec))
                extra_bits = []
                if rec.get("ROR") is not None:
                    extra_bits.append(f"ROR = {rec['ROR']}")
                if rec.get("Chi-square") is not None:
                    extra_bits.append(f"chi-square = {rec['Chi-square']}")
                if rec.get("ML Anomaly"):
                    extra_bits.append(str(rec["ML Anomaly"]))
                if rec.get("Reviewer Priority Score") is not None:
                    extra_bits.append(f"priority score = {rec['Reviewer Priority Score']}")
                if extra_bits:
                    st.caption("Additional screening metrics: " + "; ".join(extra_bits))

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
    st.title("Submission Readiness")
    st.caption("Representative ICH M4 CTD completeness checker — not a compliance certification.")
    _disclaimer()
    st.info(
        "Representative prototype checklist — not an exhaustive regulatory submission checklist."
    )

    with st.expander("Representative checklist used by this prototype", expanded=False):
        for module, spec in REPRESENTATIVE_CHECKLIST.items():
            st.markdown(f"**{module}** (weight {int(spec['weight'] * 100)}%)")
            st.write(", ".join(section for section, _priority in spec["sections"]))

    source_choice = st.radio(
        "Outline source",
        ["Sample / Synthetic Data", "Upload or paste"],
        horizontal=True,
    )
    uploaded = None
    paste = ""
    if source_choice == "Upload or paste":
        uploaded = st.file_uploader(
            "Upload dossier outline (.txt or .csv)",
            type=["txt", "csv"],
        )
        paste = st.text_area(
            "Or paste dossier sections (one per line)",
            height=180,
            placeholder="Module 3: Drug Substance\nModule 5: Clinical Safety Study Reports",
        )
    else:
        st.caption("Using bundled Sample / Synthetic Data (not a real dossier).")

    try:
        if source_choice == "Sample / Synthetic Data":
            text = SAMPLE_DOSSIER_OUTLINE.read_text(encoding="utf-8")
            source_label = "Sample / Synthetic Data"
        elif uploaded is not None:
            text = uploaded.getvalue().decode("utf-8")
            source_label = "Uploaded file"
        elif paste.strip():
            text = paste
            source_label = "Pasted text"
        else:
            st.info("Upload, paste, or switch Outline source to Sample / Synthetic Data.")
            return
        result = run_dossier_pipeline(text)
    except Exception as exc:  # noqa: BLE001
        _show_error(exc)
        return

    st.success(
        f"Outline source: **{source_label}** · parsed {result['provided_count']} section line(s)"
    )
    if source_label.startswith("Sample"):
        st.warning("This outline is **Sample / Synthetic Data**, not a real regulatory dossier.")

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Overall readiness score", f"{result['overall_score']}")
    s2.metric("Present sections", f"{result['present_total']} / {result['expected_total']}")
    s3.metric("Missing sections", result["missing_total"])
    s4.metric("High-priority gaps", result["missing_high"])
    st.progress(min(max(result["overall_score"] / 100.0, 0.0), 1.0))
    st.caption(result["score_formula"])

    st.subheader("Module scores")
    module_cols = st.columns(5)
    for idx, rec in enumerate(result["module_scores"].to_dict(orient="records")):
        with module_cols[idx]:
            st.metric(_module_label(rec["Module"]), f"{rec['Completeness %']}%")
            st.progress(min(max(rec["Completeness %"] / 100.0, 0.0), 1.0))

    if result["present_total"] == 0:
        st.warning(
            "⚠️ **0 recognized dossier sections found.** None of the lines in this file matched the representative ICH M4 CTD sections. "
            "If you uploaded an adverse-event file (e.g., `adverse_events.csv`), switch to **Mode 1 — Signal Detection** in the sidebar. "
            "For Submission Readiness, use a dossier outline file like `sample_data/demo_video_dossier.csv` or choose **Sample / Synthetic Data** above."
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
    st.subheader("Gap report")
    st.dataframe(gap, use_container_width=True)

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
            st.dataframe(subset, use_container_width=True)
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
        help="Module | Section | Status | Priority | Recommendation",
    )


def main() -> None:
    st.sidebar.title(TEAM_NAME)
    st.sidebar.caption("AI Track prototype")
    page = st.sidebar.selectbox(
        "Navigate",
        ["Home", "Signal Detection", "Submission Readiness"],
    )
    st.sidebar.markdown("---")
    st.sidebar.write(
        "PRR is a statistical screening measure. CTD scoring uses a representative "
        "checklist. Expert review is required."
    )
    if page == "Home":
        render_home()
    elif page == "Signal Detection":
        render_signal_detection()
    else:
        render_submission()


if __name__ == "__main__":
    main()
