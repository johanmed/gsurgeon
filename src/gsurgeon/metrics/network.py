"""Module with constructs to compute metrics for network analysis"""

import dspy

from gsurgeon.metrics.bootstrapping import bootstrap


class ExtractEdge(dspy.Signature):
    """
    Construct a list of connections consistent across answers.
    Scan answers and extract weight for each connection in the previous list.
    Build a dictionary with edge as key and list of weights as value.
    Example: {"gene A -> gene C": [1, 1, 2], "gene B -> gene D": [2, 3, 2]}
    """

    query: str = dspy.InputField(desc="Network analysis query")
    answers: list[str] = dspy.InputField(
        desc="List of answers to the query, each reporting a list of weighted connections between genes"
    )
    edge_weights: dict[str, list[int]] = dspy.OutputField(
        desc="Dictionary of connections and assigned weights"
    )


def bootstrap_weight(
    query: str, answers: list[str]
) -> dict[tuple[str, int], tuple[float, int]]:
    """Compute a bootstrap weight for each edge"""
    extract = dspy.Predict(ExtractEdge)
    edge_weights = extract(query=query, answers=answers).get("edge_weights")
    return bootstrap(edge_weights, descend=True)
