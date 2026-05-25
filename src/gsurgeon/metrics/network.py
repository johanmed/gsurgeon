"""Module with constructs to compute metrics for network analysis"""

import dspy
from gsurgeon.metrics.bootstrapping import bootstrap


class ExtractEdge(dspy.Signature):
    """
    Construct a list of connections consistent across answers.
    Scan answers and extract rank for each connection in the previous list.
    Build a dictionary with edge as key and list of ranks as value.
    Example: {"gene A -> gene C": [1, 1, 2], "gene B -> gene D": [2, 3, 2]}
    """

    query: str = dspy.InputField(desc="Network analysis query")
    answers: list[str] = dspy.InputField(
        desc="List of answers to the query, each reporting a ranked of list of connections between genes"
    )
    edge_ranks: dict[str, list] = dspy.OutputField(
        desc="Dictionary of connections and assigned ranks assigned"
    )


def bootstrap_rank(query: str, answers: list[str]) -> dict[tuple, float]:
    """Compute a bootstrap rank for each edge"""
    extract = dspy.Predict(ExtractEdge)
    edge_ranks = extract(query=query, answers=answers).get("edge_ranks")
    return bootstrap(edge_ranks)
