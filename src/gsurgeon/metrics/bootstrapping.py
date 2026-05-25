"""Module with bootstrapping logic"""

from collections import Counter


def bootstrap(element_ranks: dict[str, list]) -> dict[tuple, float]:
    """
    Compute bootstrap rank for each element
    Args:
        element_ranks: dictionary of elements and ranks (list)
    Output:
        Dictionary of elements and bootstrap ranks (single value)
    """
    bootstrap_ranks = {}
    for element, ranks in element_ranks.items():
        sorted_ranks = sorted(ranks)
        total = len(sorted_ranks)
        top = sorted(
            Counter(sorted_ranks).items(), key=lambda item: item[1], reverse=True
        )[0]
        top_rank, frequency = top
        bootstraps[(element, top_rank)] = frequency / total
    return bootstrap_ranks
