"""
Script to run gsurgeon with bootstrapping on all questions in a set
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
        max_tokens=10_000,
        temperature=0,
        cache=False,
        verbose=False,
    )

    dspy.configure(lm=model)

    N_ITERATIONS = os.getenv("N_ITERATIONS")
    if N_ITERATIONS is None:
        raise ValueError("Set N_ITERATIONS for operation")
    n_iterations = int(N_ITERATIONS)

    N_BOOTSTRAPS = os.getenv("N_BOOTSTRAPS")
    if N_BOOTSTRAPS is None:
        raise ValueError("Set N_BOOTSTRAPS for multiple runs")
    n_bootstraps = int(N_BOOTSTRAPS)

    print("Running operation for a set of queries...")
    with open(args.input_path) as i:
        data = i.read()
        queries = data.strip().split("\n")
    collection = asyncio.run(serialize(queries, n_iterations, n_bootstraps))
    with open(args.output_path, "w") as o:
        o.write(json.dumps(collection, indent=4))
    print("Run complete for all queries")
