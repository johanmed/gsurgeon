"""
Script to run gsurgeon with repetition on all questions in a set
Author: Johannes Medagbe
Copyright (c) 2026
"""

import argparse
import asyncio
import json
import os

import dspy
from Bio import Entrez
from dotenv import load_dotenv
from gsurgeon.operations.standard import serialize

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", help="Path to file with set of questions")
    parser.add_argument("--output-path", help="Path to output file")
    parser.add_argument("--env-file", default=".env", help="Path to .env file")
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
        max_tokens=20_000,
        temperature=0,
        cache=False,
        verbose=False,
    )

    dspy.configure(lm=model)

    N_STEPS = os.getenv("N_STEPS")
    if N_STEPS is None:
        raise ValueError("Set N_STEPS for operation")
    n_steps = int(N_STEPS)

    N_ITERATIONS = os.getenv("N_ITERATIONS")
    if N_ITERATIONS is None:
        raise ValueError("Set N_ITERATIONS for multiple runs")
    n_iterations = int(N_ITERATIONS)

    print("Running operation for a set of queries...")
    with open(args.input_path) as i:
        data = i.read()
        queries = data.strip().split("\n")
    collection = asyncio.run(serialize(queries, n_steps, n_iterations))
    with open(args.output_path, "w") as o:
        o.write(json.dumps(collection, indent=4))
    print("Run complete for all queries")
