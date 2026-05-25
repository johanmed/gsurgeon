"""Module with standard operation constructs for GSurgeon"""

import asyncio

import dspy
from gsurgeon.procedures.standard import Reproduce
from gsurgeon.surgeon.agent import GSurgeon


async def operate(query: str, n_iterations: int = 5) -> str:
    """Execute operation or analysis with GSurgeon"""
    surgeon = GSurgeon(max_iterations=n_iterations)
    return await surgeon.handle(query)


async def reoperate(query: str, n_iterations: int = 5, n_bootstraps: int = 5) -> str:
    """
    Execute operation or analysis a given number of times for reproducibility
    Args:
        query: inquiry
        n_iterations: max number of iterations allowed during operation
        n_boostraps: number of operation repetitions
    Output:
        Consensus resulting from different runs
    """
    print(f"Bootstrapping operation {n_bootstraps} times for query...")
    results = await asyncio.gather(
        *[operate(query, n_iterations) for n in range(n_bootstraps)]
    )
    reproduce = dspy.Predict(Reproduce)
    print("Bootstrapped run completed")
    return reproduce(query=query, results=results).get("consensus")


async def serialize(
    queries: list, n_iterations: int = 5, n_bootstraps: int = 5
) -> dict:
    """
    Execute operation a given number of times for a set/series of queries
    Args:
        queries: list of queries to investigate
        n_iterations: max number of iterations allowed during operation
        n_boostraps: number of operation repetitions
    Output:
        Dictionary of query and response
    """
    results = await asyncio.gather(
        *[reoperate(query, n_iterations, n_bootstraps) for query in queries]
    )
    collection = {}
    for query, result in zip(queries, results):
        collection[f"Query was '{query}'"] = f"Response was '{result}'."
    return collection
