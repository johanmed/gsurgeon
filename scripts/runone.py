#!/usr/bin/env python3

import argparse
import asyncio
import os

import dspy
from Bio import Entrez
from dotenv import load_dotenv

from gsurgeon.agent import GSurgeon


async def operate(query: str, n_iterations: int = 5) -> str:
    surgeon = GSurgeon(max_iterations=n_iterations)
    return await surgeon.handle(query)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", default=".env", help="Path to .env file")
    parser.add_argument("query", help="Question to answer")
    args = parser.parse_args()

    load_dotenv(dotenv_path=args.env_file)

    EMAIL = os.getenv("EMAIL")
    if EMAIL is None:
        raise ValueError("Set EMAIL for NCBI tool calling")
    Entrez.email = EMAIL

    MODEL_NAME = os.getenv("MODEL_NAME")
    if MODEL_NAME is None:
        raise ValueError("Set MODEL_NAME of a provider")

    API_KEY = os.getenv("API_KEY")
    if API_KEY is None:
        raise ValueError("Set valid API_KEY for proprietary model")
    model = dspy.LM(
        MODEL_NAME,
        api_key=API_KEY,
        max_tokens=10_000,
        temperature=0,
        verbose=False,
    )

    dspy.configure(lm=model)

    N_ITERATIONS = os.getenv("N_ITERATIONS")
    if N_ITERATIONS is None:
        raise ValueError("Set N_ITERATIONS for operation")
    n_iterations = int(N_ITERATIONS)

    print(asyncio.run(operate(args.query, n_iterations)))
