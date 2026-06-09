"""Module with standard operation constructs for GSurgeon"""

import asyncio

import dspy
from gsurgeon.procedures.standard import Reproduce
from gsurgeon.surgeon.agent import GSurgeon


async def operate(query: str, n_steps: int = 5) -> str:
    """Execute operation or analysis with GSurgeon"""
    surgeon = GSurgeon(max_steps=n_steps)
    return await surgeon.handle(query)


async def reoperate(query: str, n_steps: int = 5, n_iterations: int = 5) -> str:
    """
    Execute operation or analysis a given number of times for reproducibility
    Args:
        query: inquiry
        n_steps: max number of steps allowed during operation
        n_iterations: number of operation repetitions
    Output:
        Consensus resulting from different runs
    """
    print(f"Repeating operation {n_iterations} times for query...")
    results = await asyncio.gather(
        *[operate(query, n_steps) for n in range(n_iterations)]
    )
    reproduce = dspy.Predict(Reproduce)
    print("Multiple run completed")
    return reproduce(query=query, results=results).get("consensus")


async def serialize(
    queries: list, n_steps: int = 5, n_iterations: int = 5
) -> dict:
    """
    Execute operation a given number of times for a set/series of queries
    Args:
        queries: list of queries to investigate
        n_steps: max number of steps allowed during operation
        n_iterations: number of operation repetitions
    Output:
        Dictionary of query and responses
    """

    base = dspy.ChainOfThought("query -> answer: str")
    base_results = [base(query=query).get("answer") for query in queries]

    surgeon_results = await asyncio.gather(
        *[reoperate(query, n_steps, n_iterations) for query in queries]
    )

    collection = {}
    for query, base_result, surgeon_result in zip(
        queries, base_results, surgeon_results
    ):
        collection[f"Query was '{query}'"] = [
            f"Base response was '{base_result}'",
            f"Surgeon response was: '{surgeon_result}'",
        ]

    return collection


async def meta_analyze(
    query: str, n_steps: int = 5, n_iterations: int = 5, n_bootstraps: int = 10
) -> list[str]:
    """
    Perform a meta-analysis of an operation or genomic task with n samples
    Args:
        query: genomic task
        n_steps: max number of steps allowed during operation
        n_iterations: number of operation repetitions
        n_bootstraps: number of repetition sampling for statistical support
    """
    return await asyncio.gather(
        *[reoperate(query, n_steps, n_iterations) for n in range(n_bootstraps)]
    )
