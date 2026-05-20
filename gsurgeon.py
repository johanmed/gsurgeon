#!/usr/bin/env python3

import asyncio
import os

import dspy
import torch

QUERY = os.getenv("QUERY")
if QUERY is None:
    raise ValueError("QUERY must be specified for program to run")

SEED = int(os.getenv("SEED"))
if SEED is None:
    raise ValueError("SEED must be specified for reproducibility")
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

EMAIL = os.getenv("EMAIL")
if EMAIL is None:
    raise ValueError("EMAIL must be specified for NCBI tool calling")
Entrez.email = EMAIL

MODEL_TYPE = int(os.getenv("MODEL_TYPE"))
if MODEL_TYPE is None:
    raise ValueError("MODEL_TYPE must be specified")

MODEL_NAME = os.getenv("MODEL_NAME")
if MODEL_NAME is None:
    raise ValueError("MODEL_NAME must be specified - either proprietary or local")

if MODEL_TYPE == 0:
    model = dspy.LM(
        model=f"openai/{MODEL_NAME}",
        api_base="http://localhost:7501/v1",
        api_key="local",
        model_type="chat",
        max_tokens=10_000,
        n_ctx=30_000,
        seed=2_025,
        temperature=0,
        verbose=False,
    )
elif MODEL_TYPE == 1:
    API_KEY = os.getenv("API_KEY")
    if API_KEY is None:
        raise ValueError("Valid API_KEY must be specified to use the proprietary model")
    model = dspy.LM(
        MODEL_NAME,
        api_key=API_KEY,
        max_tokens=10_000,
        temperature=0,
        verbose=False,
    )
else:
    raise ValueError("MODEL_TYPE must be 0 or 1")

dspy.configure(lm=model)


async def operate(query: str) -> str:
    surgeon = GSurgeon()
    return await surgeon.handler(query)


if __name__ == "__main__":
    asyncio.run(operate(QUERY))
