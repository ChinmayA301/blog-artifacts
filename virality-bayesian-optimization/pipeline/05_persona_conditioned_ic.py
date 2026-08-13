#!/usr/bin/env python3
"""Approach B: topic-conditional influence maximization.

Provenance is deliberately split into three layers:

1. real topology: the SNAP Higgs retweet graph;
2. synthetic node attributes: MatrAIx personas are assigned to anonymous graph
   nodes under an explicit, sensitivity-tested conditioning rule; and
3. real method: heterogeneous Independent Cascade simulation plus Bayesian
   Optimization over an interpretable seed-selection rule.

The persona-to-node join is not observed data. Even a human-grounded MatrAIx
row becomes synthetic metadata once assigned to an anonymous Higgs node. Run
``audit`` before ``experiment`` and report the coupling sensitivity, not only
the best-performing setting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import networkx as nx
import numpy as np
import pandas as pd


PUBLIC_SAMPLE_URL = (
    "https://huggingface.co/datasets/MatrAIx2026/"
    "MatrAIx_Persona_1M_Public_Release/resolve/main/sample/sample.parquet"
)
PUBLIC_SCHEMA_URL = (
    "https://huggingface.co/datasets/MatrAIx2026/"
    "MatrAIx_Persona_1M_Public_Release/resolve/main/persona_codes.schema.json"
)

REQUIRED_COLUMNS = [
    "source",
    "age_bracket",
    "region",
    "tech_savviness",
    "topic_technology",
    "topic_science",
    "fam_machine_learning",
    "topic_sports",
    "ind_sports",
    "att_social_media",
    "att_influencers",
    "bfi2_domain_extraversion",
    "big5_gregariousness",
    "bfi2_facet_sociability",
    "big5_excitement_seeking",
    "trait_curiosity",
    "need_for_cognition",
    "cog_reading_vs_watching",
    "lstyle_news_freq",
]

ORDINAL_MAPS: dict[str, dict[str, float]] = {
    "tech_savviness": {
        "Avoidant": 0.0,
        "Reluctant": 0.2,
        "Cautious adopter": 0.45,
        "Comfortable": 0.72,
        "Digital native": 1.0,
    },
    "interest": {
        "Averse": 0.0,
        "Indifferent": 0.2,
        "Neutral": 0.45,
        "Interested": 0.75,
        "Passionate": 1.0,
    },
    "familiarity": {
        "None": 0.0,
        "Aware": 0.25,
        "Familiar": 0.55,
        "Proficient": 0.8,
        "Expert": 1.0,
    },
    "industry": {
        "None": 0.0,
        "Some exposure": 0.35,
        "Experienced": 0.72,
        "Veteran": 1.0,
    },
    "attitude": {
        "Opposed": 0.0,
        "Skeptical": 0.2,
        "Neutral": 0.48,
        "Positive": 0.75,
        "Enthusiast": 1.0,
    },
    "five_point": {
        "Very low": 0.0,
        "Low": 0.25,
        "Average": 0.5,
        "Moderate": 0.5,
        "High": 0.75,
        "Very high": 1.0,
    },
    "character_strength": {
        "Absent": 0.0,
        "Slight": 0.25,
        "Moderate": 0.5,
        "Strong": 0.75,
        "Signature": 1.0,
    },
    "reading_preference": {
        "Strongly prefers video": 0.0,
        "Prefers video": 0.25,
        "No preference": 0.5,
        "Prefers reading": 0.75,
        "Strongly prefers reading": 1.0,
    },
    "news_frequency": {
        "Avoids news": 0.0,
        "Rarely": 0.2,
        "Weekly": 0.5,
        "Daily": 0.8,
        "Constant": 1.0,
    },
}

FIELD_MAP = {
    "tech_savviness": "tech_savviness",
    "topic_technology": "interest",
    "topic_science": "interest",
    "fam_machine_learning": "familiarity",
    "topic_sports": "interest",
    "ind_sports": "industry",
    "att_social_media": "attitude",
    "att_influencers": "attitude",
    "bfi2_domain_extraversion": "five_point",
    "big5_gregariousness": "five_point",
    "bfi2_facet_sociability": "five_point",
    "big5_excitement_seeking": "five_point",
    "trait_curiosity": "character_strength",
    "need_for_cognition": "five_point",
    "cog_reading_vs_watching": "reading_preference",
    "lstyle_news_freq": "news_frequency",
}

AGE_MIN = {
    "Under 5": 0,
    "5-12": 5,
    "13-17": 13,
    "18-24": 18,
    "25-34": 25,
    "35-44": 35,
    "45-54": 45,
    "55-64": 55,
    "65-74": 65,
    "75-84": 75,
    "85+": 85,
}


@dataclass(frozen=True)
class TopicProfile:
    name: str
    affinity_column: str


TOPICS = {
    "ml_paper": TopicProfile("ml_paper", "affinity_ml_paper"),
    "sports_clip": TopicProfile("sports_clip", "affinity_sports_clip"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _numeric_field(frame: pd.DataFrame, field: str) -> pd.Series:
    mapping = ORDINAL_MAPS[FIELD_MAP[field]]
    if field not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    clean = frame[field].replace({"null": np.nan, "None": np.nan, "": np.nan})
    return clean.map(mapping).astype(float)


def _weighted_mean(
    numeric: Mapping[str, pd.Series], weights: Mapping[str, float]
) -> pd.Series:
    fields = list(weights)
    values = np.column_stack([numeric[name].to_numpy(float) for name in fields])
    raw_weights = np.asarray([weights[name] for name in fields], dtype=float)
    present = np.isfinite(values)
    numerator = np.nansum(values * raw_weights, axis=1)
    denominator = np.sum(present * raw_weights, axis=1)
    result = np.divide(
        numerator,
        denominator,
        out=np.full(len(values), np.nan),
        where=denominator > 0,
    )
    return pd.Series(result, index=numeric[fields[0]].index)


def load_persona_signals(path: Path) -> tuple[pd.DataFrame, dict]:
    """Read the decoded public MatrAIx Parquet sample and derive model signals."""
    frame = pd.read_parquet(path, columns=REQUIRED_COLUMNS)
    numeric = {field: _numeric_field(frame, field) for field in FIELD_MAP}

    signals = pd.DataFrame(index=frame.index)
    signals["source"] = frame["source"].fillna("unknown")
    signals["age_bracket"] = frame["age_bracket"]
    signals["region"] = frame["region"]
    signals["share_propensity"] = _weighted_mean(
        numeric,
        {
            "att_social_media": 0.24,
            "att_influencers": 0.12,
            "bfi2_domain_extraversion": 0.16,
            "big5_gregariousness": 0.16,
            "bfi2_facet_sociability": 0.16,
            "lstyle_news_freq": 0.16,
        },
    )
    signals["affinity_ml_paper"] = _weighted_mean(
        numeric,
        {
            "fam_machine_learning": 0.26,
            "topic_technology": 0.20,
            "topic_science": 0.18,
            "tech_savviness": 0.14,
            "need_for_cognition": 0.12,
            "cog_reading_vs_watching": 0.10,
        },
    )
    video_preference = 1.0 - numeric["cog_reading_vs_watching"]
    sports_numeric = dict(numeric)
    sports_numeric["video_preference"] = video_preference
    signals["affinity_sports_clip"] = _weighted_mean(
        sports_numeric,
        {
            "topic_sports": 0.42,
            "ind_sports": 0.22,
            "video_preference": 0.14,
            "big5_excitement_seeking": 0.12,
            "att_social_media": 0.10,
        },
    )

    for column in ["share_propensity", "affinity_ml_paper", "affinity_sports_clip"]:
        median = float(signals[column].median())
        signals[column] = signals[column].fillna(median).clip(0.0, 1.0)

    min_age = frame["age_bracket"].map(AGE_MIN)
    signals["age_eligible"] = min_age.ge(13).fillna(False)
    signals["platform_fit"] = (
        0.7 * signals["share_propensity"]
        + 0.2 * numeric["tech_savviness"].fillna(0.5)
        + 0.1 * numeric["lstyle_news_freq"].fillna(0.5)
    ).clip(0.0, 1.0)

    # Conditioning is deliberately soft after the hard age gate. It is a
    # modeling prior, not an estimate of Twitter's 2012 user distribution.
    raw_weight = np.where(
        signals["age_eligible"],
        0.15 + 0.85 * signals["platform_fit"].to_numpy(),
        0.0,
    )
    if raw_weight.sum() == 0:
        raise ValueError("No age-eligible persona rows remain after conditioning")
    signals["sampling_weight"] = raw_weight / raw_weight.sum()

    coverage = {
        field: float(
            frame[field]
            .replace({"null": np.nan, "None": np.nan, "": np.nan})
            .notna()
            .mean()
        )
        for field in FIELD_MAP
    }
    audit = {
        "rows": int(len(frame)),
        "file_sha256": sha256(path),
        "source_distribution": {
            str(key): int(value)
            for key, value in frame["source"].fillna("unknown").value_counts().items()
        },
        "required_field_coverage": coverage,
        "age_eligible_rows": int(signals["age_eligible"].sum()),
        "age_eligible_share": float(signals["age_eligible"].mean()),
        "effective_sample_size_after_conditioning": float(
            1.0 / np.square(signals["sampling_weight"]).sum()
        ),
        "topic_signal_summary": {
            topic: {
                "mean": float(signals[profile.affinity_column].mean()),
                "std": float(signals[profile.affinity_column].std(ddof=0)),
            }
            for topic, profile in TOPICS.items()
        },
        "topic_affinity_correlation": float(
            signals["affinity_ml_paper"].corr(signals["affinity_sports_clip"])
        ),
    }
    return signals, audit


def _zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    std = values.std()
    return (values - values.mean()) / std if std > 0 else np.zeros_like(values)


def _rank_correlation(x: np.ndarray, y: np.ndarray) -> float:
    xr = pd.Series(x).rank(method="average").to_numpy()
    yr = pd.Series(y).rank(method="average").to_numpy()
    return float(np.corrcoef(xr, yr)[0, 1])


def assign_personas_to_nodes(
    graph: nx.DiGraph,
    signals: pd.DataFrame,
    coupling: float,
    rng: np.random.Generator,
) -> tuple[list, pd.DataFrame, dict]:
    """Assign sampled persona rows to nodes using a transparent copula prior.

    ``coupling=0`` is random assignment. Positive values increasingly align a
    node's observed activity rank with the sampled persona's platform-fit rank.
    No value is asserted to be true; the experiment must be reported across a
    sensitivity grid.
    """
    if not 0.0 <= coupling <= 0.95:
        raise ValueError("coupling must be in [0, 0.95]")
    nodes = list(graph.nodes())
    probabilities = signals["sampling_weight"].to_numpy(float)
    draws = rng.choice(len(signals), size=len(nodes), replace=True, p=probabilities)
    sampled = signals.iloc[draws].reset_index(drop=True)

    activity = np.array(
        [math.log1p(graph.out_degree(n)) + 0.5 * math.log1p(graph.in_degree(n)) for n in nodes],
        dtype=float,
    )
    latent = coupling * _zscore(activity) + math.sqrt(1.0 - coupling**2) * rng.normal(
        size=len(nodes)
    )
    node_order = np.argsort(latent)
    persona_order = np.argsort(sampled["platform_fit"].to_numpy())
    assignment_order = np.empty(len(nodes), dtype=int)
    assignment_order[node_order] = persona_order
    assigned = sampled.iloc[assignment_order].reset_index(drop=True)

    diagnostics = {
        "requested_coupling": coupling,
        "realized_spearman_activity_platform_fit": _rank_correlation(
            activity, assigned["platform_fit"].to_numpy()
        ),
        "unique_persona_rows_used": int(len(np.unique(draws))),
        "persona_reuse_rate": float(1.0 - len(np.unique(draws)) / len(nodes)),
    }
    return nodes, assigned, diagnostics


def _logit(probability: float) -> float:
    return math.log(probability / (1.0 - probability))


def _sigmoid(value: float) -> float:
    if value >= 0:
        exp_neg = math.exp(-value)
        return 1.0 / (1.0 + exp_neg)
    exp_pos = math.exp(value)
    return exp_pos / (1.0 + exp_pos)


def activation_probability(
    sender: Mapping[str, float],
    receiver: Mapping[str, float],
    topic: TopicProfile,
    base_p: float = 0.12,
    beta_affinity: float = 1.1,
    beta_share: float = 0.7,
    beta_homophily: float = 0.35,
) -> float:
    """Topic-conditional edge probability, clipped away from 0 and 1."""
    affinity_u = float(sender[topic.affinity_column])
    affinity_v = float(receiver[topic.affinity_column])
    share_v = float(receiver["share_propensity"])
    homophily = 1.0 - abs(affinity_u - affinity_v)
    linear = (
        _logit(base_p)
        + beta_affinity * (affinity_v - 0.5) * 2.0
        + beta_share * (share_v - 0.5) * 2.0
        + beta_homophily * (homophily - 0.5) * 2.0
    )
    return float(np.clip(_sigmoid(linear), 0.005, 0.60))


def conditional_ic(
    graph: nx.DiGraph,
    seeds: Sequence,
    nodes: Sequence,
    assigned: pd.DataFrame,
    topic: TopicProfile,
    n_sims: int,
    rng: np.random.Generator,
    base_p: float = 0.12,
) -> float:
    node_index = {node: index for index, node in enumerate(nodes)}
    records = assigned.to_dict("records")
    total = 0
    for _ in range(n_sims):
        active = set(seeds)
        frontier = list(seeds)
        while frontier:
            following = []
            for sender_node in frontier:
                sender = records[node_index[sender_node]]
                for receiver_node in graph.successors(sender_node):
                    if receiver_node in active:
                        continue
                    receiver = records[node_index[receiver_node]]
                    probability = activation_probability(sender, receiver, topic, base_p=base_p)
                    if rng.random() < probability:
                        active.add(receiver_node)
                        following.append(receiver_node)
            frontier = following
        total += len(active)
    return total / n_sims


def build_topic_seeds(
    graph: nx.DiGraph,
    nodes: Sequence,
    assigned: pd.DataFrame,
    topic: TopicProfile,
    theta: Sequence[float],
    k: int,
    pool_size: int,
) -> list:
    degree = np.asarray([graph.out_degree(node) for node in nodes], dtype=float)
    pool_size = min(pool_size, len(nodes))
    pool_index = np.argpartition(-degree, pool_size - 1)[:pool_size]
    pool_nodes = [nodes[index] for index in pool_index]
    pool_degree = _zscore(np.log1p(degree[pool_index]))
    affinity = _zscore(assigned.iloc[pool_index][topic.affinity_column].to_numpy())
    sharing = _zscore(assigned.iloc[pool_index]["share_propensity"].to_numpy())
    w_degree, w_topic, w_share, redundancy = map(float, theta)
    base_score = w_degree * pool_degree + w_topic * affinity + w_share * sharing

    pool_subgraph = graph.subgraph(pool_nodes).to_undirected()
    neighborhoods = {node: set(pool_subgraph.neighbors(node)) for node in pool_nodes}
    available = np.ones(pool_size, dtype=bool)
    chosen: list = []
    covered: set = set()
    for _ in range(k):
        score = base_score.copy()
        if chosen and redundancy > 0:
            overlap = np.asarray(
                [1.0 if available[i] and node in covered else 0.0 for i, node in enumerate(pool_nodes)]
            )
            score -= redundancy * overlap
        score[~available] = -np.inf
        selected_index = int(np.argmax(score))
        selected_node = pool_nodes[selected_index]
        chosen.append(selected_node)
        available[selected_index] = False
        covered.update(neighborhoods[selected_node])
    return chosen


def run_topic_bo(
    graph: nx.DiGraph,
    nodes: Sequence,
    assigned: pd.DataFrame,
    topic: TopicProfile,
    rng: np.random.Generator,
    k: int,
    pool_size: int,
    evaluations: int,
    mc: int,
) -> dict:
    from scipy.stats import norm
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

    bounds = np.array([[0.1, 2.2], [0.0, 2.2], [0.0, 1.6], [0.0, 4.0]])

    def sample(count: int) -> np.ndarray:
        return rng.uniform(bounds[:, 0], bounds[:, 1], size=(count, len(bounds)))

    evaluation_counter = 0

    def objective(theta: np.ndarray) -> float:
        nonlocal evaluation_counter
        seeds = build_topic_seeds(graph, nodes, assigned, topic, theta, k, pool_size)
        result = conditional_ic(
            graph,
            seeds,
            nodes,
            assigned,
            topic,
            n_sims=mc,
            rng=np.random.default_rng(10_000 + evaluation_counter),
        )
        evaluation_counter += 1
        return result

    initial = min(8, evaluations)
    x_values = sample(initial)
    y_values = np.asarray([objective(theta) for theta in x_values])
    curve = [float(y_values.max())]
    while len(y_values) < evaluations:
        kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(4), nu=2.5) + WhiteKernel(
            noise_level=4.0
        )
        gp = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=1,
            random_state=17,
        ).fit(x_values, y_values)
        candidates = sample(1200)
        mean, std = gp.predict(candidates, return_std=True)
        std = np.maximum(std, 1e-9)
        improvement = mean - y_values.max() - 0.01
        z_value = improvement / std
        expected_improvement = improvement * norm.cdf(z_value) + std * norm.pdf(z_value)
        next_theta = candidates[int(np.argmax(expected_improvement))]
        next_value = objective(next_theta)
        x_values = np.vstack([x_values, next_theta])
        y_values = np.append(y_values, next_value)
        curve.append(float(y_values.max()))

    best_theta = x_values[int(np.argmax(y_values))]
    best_seeds = build_topic_seeds(
        graph, nodes, assigned, topic, best_theta, k=k, pool_size=pool_size
    )
    final_spread = conditional_ic(
        graph,
        best_seeds,
        nodes,
        assigned,
        topic,
        n_sims=max(200, mc * 4),
        rng=np.random.default_rng(99_001),
    )
    top_degree = sorted(nodes, key=graph.out_degree, reverse=True)[:k]
    top_degree_spread = conditional_ic(
        graph,
        top_degree,
        nodes,
        assigned,
        topic,
        n_sims=max(200, mc * 4),
        rng=np.random.default_rng(99_002),
    )
    return {
        "topic": topic.name,
        "best_theta": [float(value) for value in best_theta],
        "bo_spread": float(final_spread),
        "top_degree_spread": float(top_degree_spread),
        "lift_vs_top_degree_pct": float(100.0 * (final_spread - top_degree_spread) / top_degree_spread),
        "best_so_far": curve,
        "bo_seeds": [str(node) for node in best_seeds],
        "top_degree_seeds": [str(node) for node in top_degree],
    }


def audit_personas(persona_path: Path, output_path: Path) -> dict:
    signals, audit = load_persona_signals(persona_path)
    audit.update(
        {
            "artifact_layer": "MatrAIx public sample; mixed grounded/synthetic persona records",
            "node_attribute_layer": "synthetic once assigned to anonymous Higgs nodes",
            "public_sample_url": PUBLIC_SAMPLE_URL,
            "public_schema_url": PUBLIC_SCHEMA_URL,
            "go_no_go": {
                "synthetic_sensitivity_experiment": "GO",
                "empirical_claim_about_2012_higgs_users": "NO-GO",
            },
            "reason": (
                "The public sample exposes usable topic, literacy, social-attitude, and "
                "personality fields, but it is not a representative sample of Twitter users "
                "during the 2012 Higgs event and contains no identity join to graph nodes."
            ),
            "required_sensitivity_grid": [0.0, 0.35, 0.7],
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return audit


def run_experiment(args: argparse.Namespace) -> dict:
    graph = pickle.loads(args.graph.read_bytes())
    if not isinstance(graph, nx.DiGraph):
        graph = nx.DiGraph(graph)
    signals, persona_audit = load_persona_signals(args.personas)
    experiment = {
        "provenance": {
            "topology": "real SNAP Higgs retweet graph",
            "persona_records": "MatrAIx public sample",
            "node_attribute_assignment": "synthetic",
            "behavioral_parameters": "synthetic and sensitivity-tested",
            "method": "heterogeneous IC plus Gaussian-process BO",
        },
        "persona_audit": persona_audit,
        "graph": {"nodes": graph.number_of_nodes(), "edges": graph.number_of_edges()},
        "settings": {
            "k": args.k,
            "pool_size": args.pool_size,
            "evaluations": args.evaluations,
            "mc": args.mc,
        },
        "couplings": [],
    }
    for coupling in args.couplings:
        assignment_rng = np.random.default_rng(args.seed + int(coupling * 1000))
        nodes, assigned, diagnostics = assign_personas_to_nodes(
            graph, signals, coupling, assignment_rng
        )
        topic_results = []
        for offset, topic in enumerate(TOPICS.values()):
            topic_results.append(
                run_topic_bo(
                    graph,
                    nodes,
                    assigned,
                    topic,
                    rng=np.random.default_rng(args.seed + offset + int(coupling * 1000)),
                    k=args.k,
                    pool_size=args.pool_size,
                    evaluations=args.evaluations,
                    mc=args.mc,
                )
            )
        seed_overlap = len(
            set(topic_results[0]["bo_seeds"]) & set(topic_results[1]["bo_seeds"])
        )
        experiment["couplings"].append(
            {
                "assignment": diagnostics,
                "topic_results": topic_results,
                "topic_seed_overlap": seed_overlap,
                "topic_seed_jaccard": seed_overlap
                / len(set(topic_results[0]["bo_seeds"]) | set(topic_results[1]["bo_seeds"])),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(experiment, indent=2) + "\n", encoding="utf-8")
    return experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    audit = subparsers.add_parser("audit", help="audit MatrAIx sample suitability")
    audit.add_argument("--personas", type=Path, required=True)
    audit.add_argument(
        "--output", type=Path, default=Path("../results/persona_conditioning_sample_audit.json")
    )

    experiment = subparsers.add_parser("experiment", help="run topic-conditional BO")
    experiment.add_argument("--personas", type=Path, required=True)
    experiment.add_argument("--graph", type=Path, required=True)
    experiment.add_argument(
        "--output", type=Path, default=Path("../results/persona_conditioned_bo.json")
    )
    experiment.add_argument("--couplings", type=float, nargs="+", default=[0.0, 0.35, 0.7])
    experiment.add_argument("--k", type=int, default=20)
    experiment.add_argument("--pool-size", type=int, default=400)
    experiment.add_argument("--evaluations", type=int, default=40)
    experiment.add_argument("--mc", type=int, default=60)
    experiment.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "audit":
        result = audit_personas(args.personas, args.output)
    else:
        result = run_experiment(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
