"""Module with constructs to compute metrics for finemapping"""

import dspy
from gsurgeon.metrics.bootstrapping import bootstrap


class ExtractGene(dspy.Signature):
    """
    Construct a list of genes consistent across answers.
    Scan answers and extract rank for each gene in the previous list.
    Build a dictionary with gene as key and list of ranks as value.
    Example: {"gene A": [1, 1, 2], "gene B": [2, 3, 2]}
    """

    query: str = dspy.InputField(desc="Finemapping query")
    answers: list[str] = dspy.InputField(
        desc="List of answers to the query, each reporting ranked list of genes"
    )
    gene_ranks: dict[str, list] = dspy.OutputField(
        desc="Dictionary of genes and assigned ranks"
    )


def bootstrap_rank(query: str, answers: list[str]) -> dict[tuple, float]:
    """Compute a bootstrap rank for each gene"""
    extract = dspy.Predict(ExtractGene)
    gene_ranks = extract(query=query, answers=answers).get("gene_ranks")
    return bootstrap(gene_ranks)
