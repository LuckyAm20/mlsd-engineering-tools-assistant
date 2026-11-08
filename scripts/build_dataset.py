"""Build relevance judgments from tool-capability mapping and family-based splits."""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VALIDATION_FAMILIES = {"storage_nvme", "debug_cpu", "mon_dashboard"}
TEST_FAMILIES = {"storage_health", "network_bandwidth", "out_of_scope"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    tools = read_csv(ROOT / "data/raw/tools.csv")
    scenarios = read_csv(ROOT / "data/raw/query_scenarios.csv")
    tool_ids = {row["tool_id"] for row in tools}
    assert len(tool_ids) == len(tools)
    assert len({row["family_id"] for row in scenarios}) == len(scenarios)
    queries: list[dict[str, object]] = []
    judgments: list[dict[str, object]] = []
    for scenario in scenarios:
        family = scenario["family_id"]
        relevant = {
            2: set(filter(None, scenario["relevance_2"].split(";"))),
            1: set(filter(None, scenario["relevance_1"].split(";"))),
        }
        assert relevant[2].isdisjoint(relevant[1])
        assert relevant[2] | relevant[1] <= tool_ids
        split = (
            "test"
            if family in TEST_FAMILIES
            else "validation"
            if family in VALIDATION_FAMILIES
            else "train"
        )
        for version in range(1, 4):
            query_id = f"q{len(queries) + 1:03d}"
            queries.append(
                {
                    "query_id": query_id,
                    "family_id": family,
                    "category": scenario["category"],
                    "split": split,
                    "query_text": scenario[f"query_{version}"],
                    "origin": "author_scenario",
                }
            )
            for tool in tools:
                tool_id = tool["tool_id"]
                label = 2 if tool_id in relevant[2] else 1 if tool_id in relevant[1] else 0
                judgments.append(
                    {
                        "query_id": query_id,
                        "tool_id": tool_id,
                        "relevance": label,
                        "annotation_method": (
                            "capability_mapping_positive" if label else "not_in_capability_mapping"
                        ),
                        "reviewed_by_domain_expert": "false",
                    }
                )
    write_csv(
        ROOT / "data/processed/queries.csv",
        ["query_id", "family_id", "category", "split", "query_text", "origin"],
        queries,
    )
    write_csv(
        ROOT / "data/processed/relevance_labels.csv",
        ["query_id", "tool_id", "relevance", "annotation_method", "reviewed_by_domain_expert"],
        judgments,
    )
    print(f"Generated {len(tools)} tools, {len(queries)} queries, {len(judgments)} pairs")


if __name__ == "__main__":
    main()
