"""Module with constructs to compute metrics for finemapping"""

import dspy

from gsurgeon.metrics.bootstrapping import bootstrap


class ExtractGene(dspy.Signature):
    """
    Construct a list of genes consistent across answers.
    Scan answers and extract rank for each gene in the previous list.
    Build a dictionary with gene as key and list of ranks as value.
    """

    query: str = dspy.InputField(desc="Finemapping query")
    answers: list[str] = dspy.InputField(
        desc="List of answers to the query, each reporting ranked list of genes"
    )
    gene_ranks: dict[str, list[int]] = dspy.OutputField(
        desc="Dictionary of genes and assigned ranks. Example: {'gene A': [1, 1, 2], 'gene B': [2, 3, 2]}"
    )


def bootstrap_rank(
    query: str, answers: list[str]
) -> dict[tuple[str, int], tuple[float, int]]:
    """Compute a bootstrap rank for each gene"""
    extract = dspy.Predict(ExtractGene)
    gene_ranks = extract(query=query, answers=answers).get("gene_ranks")
    return bootstrap(gene_ranks)
