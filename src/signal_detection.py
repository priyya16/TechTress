"""Pattern detection: TF-IDF + KMeans when free text exists, else frequencies."""

from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

from data_processor import detect_text_column


def _choose_k(n_docs: int) -> int:
    if n_docs < 8:
        return 2
    return max(2, min(6, n_docs // 8))


def cluster_reports(frame: pd.DataFrame, random_state: int = 42) -> dict:
    """Cluster narratives when possible; otherwise group by adverse event.

    Returns a dictionary describing the method used, summaries, and a frame
    with an optional cluster label column.
    """
    labeled = frame.copy()
    text_col = detect_text_column(labeled)

    if text_col is None:
        counts = (
            labeled.groupby("adverse_event")
            .size()
            .reset_index(name="report_count")
            .sort_values("report_count", ascending=False)
        )
        return {
            "method": "categorical_frequency",
            "used_column": "adverse_event",
            "explanation": (
                "Unable to perform clustering because usable text data is unavailable. "
                "Pattern detection uses frequency grouping of the adverse_event "
                "category instead of TF-IDF clustering."
            ),
            "summaries": counts,
            "frame": labeled,
            "n_clusters": 0,
            "n_narrative_reports": 0,
            "silhouette": None,
            "lda_topics": pd.DataFrame(),
            "similar_reports": pd.DataFrame(),
        }

    documents = labeled[text_col].fillna("").astype(str)
    usable = documents.str.len() >= 20
    if usable.sum() < 8:
        counts = (
            labeled.groupby("adverse_event")
            .size()
            .reset_index(name="report_count")
            .sort_values("report_count", ascending=False)
        )
        return {
            "method": "categorical_frequency",
            "used_column": "adverse_event",
            "explanation": (
                "Unable to perform clustering because usable text data is unavailable. "
                f"Column '{text_col}' does not contain enough distinct narratives, "
                "so frequency grouping of adverse_event is used instead."
            ),
            "summaries": counts,
            "frame": labeled,
            "n_clusters": 0,
            "n_narrative_reports": 0,
            "silhouette": None,
            "lda_topics": pd.DataFrame(),
            "similar_reports": pd.DataFrame(),
        }

    vectorizer = TfidfVectorizer(
        max_features=500,
        stop_words="english",
        min_df=1,
        ngram_range=(1, 2),
    )
    matrix = vectorizer.fit_transform(documents.where(usable, other="event report"))
    n_clusters = _choose_k(int(usable.sum()))
    model = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    labels = model.fit_predict(matrix)
    labeled["cluster"] = labels

    terms = vectorizer.get_feature_names_out()
    summaries = []
    for cluster_id in range(n_clusters):
        center = model.cluster_centers_[cluster_id]
        top_idx = center.argsort()[::-1][:6]
        top_terms = ", ".join(terms[i] for i in top_idx)
        member_mask = labeled["cluster"] == cluster_id
        top_events = (
            labeled.loc[member_mask, "adverse_event"]
            .value_counts()
            .head(3)
            .index.tolist()
        )
        examples = (
            labeled.loc[member_mask, text_col]
            .fillna("")
            .astype(str)
            .map(str.strip)
        )
        unique_examples: list[str] = []
        for text in examples.tolist():
            if len(text) < 20 or text in unique_examples:
                continue
            unique_examples.append(text)
            if len(unique_examples) == 3:
                break
        summaries.append(
            {
                "cluster": cluster_id,
                "reports": int(member_mask.sum()),
                "top_terms": top_terms,
                "common_events": ", ".join(top_events),
                "example_narratives": unique_examples,
            }
        )

    silhouette = None
    try:
        if n_clusters >= 2 and usable.sum() >= n_clusters + 1:
            dense = matrix.toarray() if hasattr(matrix, "toarray") else matrix
            silhouette = float(silhouette_score(dense, labels, metric="cosine"))
    except ValueError:
        silhouette = None

    lda_topics = pd.DataFrame()
    try:
        count_vec = CountVectorizer(max_features=400, stop_words="english", min_df=1)
        counts = count_vec.fit_transform(documents.where(usable, other="event report"))
        lda = LatentDirichletAllocation(
            n_components=n_clusters,
            random_state=random_state,
            learning_method="batch",
        )
        lda.fit(counts)
        vocab = count_vec.get_feature_names_out()
        topic_rows = []
        for topic_id, topic in enumerate(lda.components_):
            top = topic.argsort()[::-1][:6]
            topic_rows.append(
                {
                    "topic": topic_id,
                    "top_words": ", ".join(vocab[i] for i in top),
                }
            )
        lda_topics = pd.DataFrame(topic_rows)
    except ValueError:
        lda_topics = pd.DataFrame()

    similar = pd.DataFrame()
    try:
        nn = NearestNeighbors(n_neighbors=min(3, matrix.shape[0]), metric="cosine")
        nn.fit(matrix)
        distances, indices = nn.kneighbors(matrix)
        rows = []
        seen = set()
        for i, (dist_row, idx_row) in enumerate(zip(distances, indices)):
            for dist, j in zip(dist_row[1:], idx_row[1:]):
                key = tuple(sorted((int(i), int(j))))
                if key in seen:
                    continue
                seen.add(key)
                sim = 1.0 - float(dist)
                if sim < 0.15:
                    continue
                left = labeled.iloc[int(i)]
                right = labeled.iloc[int(j)]
                rows.append(
                    {
                        "similarity": round(sim, 3),
                        "report_a": f"{left['drug_name']} · {left['adverse_event']}",
                        "report_b": f"{right['drug_name']} · {right['adverse_event']}",
                        "snippet_a": str(left[text_col])[:140],
                        "snippet_b": str(right[text_col])[:140],
                    }
                )
        if rows:
            similar = (
                pd.DataFrame(rows)
                .sort_values("similarity", ascending=False)
                .head(8)
                .reset_index(drop=True)
            )
    except ValueError:
        similar = pd.DataFrame()

    return {
        "method": "tfidf_kmeans",
        "used_column": text_col,
        "explanation": (
            f"TF-IDF vectors were built from '{text_col}', then KMeans found "
            f"{n_clusters} clusters. Optional LDA topics and cosine nearest-neighbor "
            "pairs are screening aids, not medical diagnoses."
        ),
        "summaries": pd.DataFrame(summaries),
        "frame": labeled,
        "n_clusters": n_clusters,
        "n_narrative_reports": int(usable.sum()),
        "silhouette": None if silhouette is None else round(silhouette, 3),
        "lda_topics": lda_topics,
        "similar_reports": similar,
    }
