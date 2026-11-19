"""Reproducible EDA for the small engineering-tools retrieval dataset."""

from pathlib import Path
from urllib.parse import urlparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGURES = ROOT / "reports" / "figures"


def get_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tools = pd.read_csv(DATA / "raw/tools.csv", keep_default_na=False)
    queries = pd.read_csv(DATA / "processed/queries.csv", keep_default_na=False)
    judgments = pd.read_csv(DATA / "processed/relevance_labels.csv", keep_default_na=False)
    return tools, queries, judgments


def validate(tools: pd.DataFrame, queries: pd.DataFrame, judgments: pd.DataFrame) -> dict[str, int]:
    """Validate structure and pair coverage without overstating label quality."""
    assert len(tools) == 18 and len(queries) == 51
    assert tools.tool_id.is_unique and queries.query_id.is_unique
    assert not judgments.duplicated(["query_id", "tool_id"]).any()
    assert set(judgments.relevance.unique()) <= {0, 1, 2}
    assert set(judgments.tool_id) == set(tools.tool_id)
    assert set(judgments.query_id) == set(queries.query_id)
    assert len(judgments) == len(tools) * len(queries)
    assert queries.groupby("family_id").split.nunique().max() == 1
    assert queries.groupby("family_id").size().eq(3).all()
    assert not queries.query_text.duplicated().any()
    for col in ["name", "category", "description_ru", "description_en", "source_url"]:
        assert tools[col].astype(str).str.strip().ne("").all(), col
    assert queries.query_text.astype(str).str.strip().ne("").all()
    assert tools.source_url.map(
        lambda url: urlparse(url).scheme == "https" and bool(urlparse(url).netloc)
    ).all()
    missing = int(
        tools.isna().sum().sum() + queries.isna().sum().sum() + judgments.isna().sum().sum()
    )
    return {
        "missing_cells": missing,
        "duplicate_tool_ids": int(tools.tool_id.duplicated().sum()),
        "duplicate_query_texts": int(queries.query_text.duplicated().sum()),
        "duplicate_pairs": int(judgments.duplicated(["query_id", "tool_id"]).sum()),
    }


def make_charts() -> dict[str, object]:
    tools, queries, judgments = get_data()
    quality = validate(tools, queries, judgments)
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "figure.dpi": 115})

    tool_counts = tools.category.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    ax.bar(tool_counts.index, tool_counts.values)
    ax.set(title="Инструменты по категориям", ylabel="Количество")
    ax.tick_params(axis="x", rotation=18)
    fig.tight_layout()
    fig.savefig(FIGURES / "01_tool_categories.png")
    plt.close(fig)

    tools = tools.assign(
        ru_words=tools.description_ru.str.split().str.len(),
        en_words=tools.description_en.str.split().str.len(),
    )
    fig, ax = plt.subplots(figsize=(8.4, 4.3))
    for category, group in tools.groupby("category", sort=True):
        ax.scatter([category] * len(group), group.ru_words, s=45, alpha=0.7)
    ax.set(title="Длина русскоязычных карточек инструментов", ylabel="Число слов")
    ax.tick_params(axis="x", rotation=18)
    fig.tight_layout()
    fig.savefig(FIGURES / "02_description_lengths.png")
    plt.close(fig)

    q_table = pd.crosstab(queries.category, queries.split).reindex(
        columns=["train", "validation", "test"], fill_value=0
    )
    fig, ax = plt.subplots(figsize=(8.4, 4.3))
    q_table.plot.bar(stacked=True, ax=ax)
    ax.set(
        title="Сценарии запросов по категориям и разбиениям",
        xlabel="Категория",
        ylabel="Количество запросов",
    )
    ax.tick_params(axis="x", rotation=18)
    fig.tight_layout()
    fig.savefig(FIGURES / "03_query_coverage.png")
    plt.close(fig)

    label_counts = judgments.relevance.value_counts().reindex([0, 1, 2], fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 4.0))
    ax.bar(["0: нет совпадения", "1: частично", "2: напрямую"], label_counts.values)
    ax.set(title="Распределение релевантности инструментов", ylabel="Число пар")
    for i, count in enumerate(label_counts.values):
        ax.text(i, count + 4, str(count), ha="center")
    ax.set_ylim(0, max(label_counts.values) * 1.16)
    fig.tight_layout()
    fig.savefig(FIGURES / "04_relevance_labels.png")
    plt.close(fig)

    positives = (
        judgments.assign(relevant=judgments.relevance > 0)
        .groupby("query_id", as_index=False)
        .relevant.sum()
    )
    positives = queries[["query_id", "split", "family_id"]].merge(positives)
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    levels = [0, 1, 2, 3]
    for split in ["train", "validation", "test"]:
        values = positives.loc[positives.split == split, "relevant"]
        ax.plot(levels, [int((values == n).sum()) for n in levels], marker="o", label=split)
    ax.set(
        xticks=levels,
        title="Число положительно размеченных инструментов на запрос",
        xlabel="Инструментов с меткой > 0",
        ylabel="Запросов",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "05_relevance_per_query.png")
    plt.close(fig)

    queries = queries.assign(words=queries.query_text.str.split().str.len())
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.hist(
        queries.words,
        bins=range(int(queries.words.min()), int(queries.words.max()) + 2),
        align="left",
        edgecolor="white",
    )
    ax.set(title="Длина пользовательских запросов", xlabel="Число слов", ylabel="Запросов")
    fig.tight_layout()
    fig.savefig(FIGURES / "06_query_lengths.png")
    plt.close(fig)

    return {
        "quality": quality,
        "tools": tools,
        "queries": queries,
        "judgments": judgments,
        "positives": positives,
        "tool_counts": tool_counts,
        "query_counts": q_table,
        "label_counts": label_counts,
    }


def main() -> None:
    results = make_charts()
    print("Generated 6 EDA charts")
    print(
        f"Tools: {len(results['tools'])}; queries: {len(results['queries'])}; "
        f"judgments: {len(results['judgments'])}"
    )
    print(f"Missing or duplicate records: {results['quality']}")


if __name__ == "__main__":
    main()
