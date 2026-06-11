"""
Script to run and bootstrap a genomic task with gsurgeon for metric estimation
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
from gsurgeon.metrics.finemapping import bootstrap_rank as rank_genes
from gsurgeon.metrics.network import bootstrap_rank as rank_edges
from gsurgeon.operations.standard import meta_analyze

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task", help="Type of genomic task to perform i.e finemapping or network"
    )
    parser.add_argument(
        "--instruction-path", help="Path to file with detailed instructions"
    )
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
        max_tokens=100_000,
        temperature=1,
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
        raise ValueError("Set N_ITERATIONS for multiple operation runs")
    n_iterations = int(N_ITERATIONS)

    N_BOOTSTRAPS = os.getenv("N_BOOTSTRAPS")
    if N_BOOTSTRAPS is None:
        raise ValueError("Set N_BOOTSTRAPS for multiple statistical runs")
    n_bootstraps = int(N_BOOTSTRAPS)

    task = args.task
    print(f"Estimating metrics for {task} task...")

    with open(args.instruction_path) as i:
        instruction = i.read().strip()
    results = asyncio.run(
        meta_analyze(instruction, n_steps, n_iterations, n_bootstraps)
    )

    if task == "finemapping":
        output = rank_genes(instruction, results)
    elif task == "network":
        output = rank_edges(instruction, results)
    else:
        raise ValueError("Genomic task not supported")
    print("Task complete and metric estimated")

    processed_output = {
        key[0]: f"Rank = {key[1]}, Bootstrap support = {round(output[key][0]*100)}%, Size = {output[key][1]}"
        for key in output
    }
    print(json.dumps(processed_output, indent=4))
