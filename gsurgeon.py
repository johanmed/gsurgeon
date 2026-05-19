import asyncio
import os
from typing import Any, Literal

import dspy
import torch

CORPUS_PATH = os.getenv("CORPUS_PATH")
if CORPUS_PATH is None:
    raise FileNotFoundError("CORPUS_PATH must be specified to find corpus")

PCORPUS_PATH = os.getenv("PCORPUS_PATH")
if PCORPUS_PATH is None:
    raise FileNotFoundError("PCORPUS_PATH must be specified to read corpus")

DB_PATH = os.getenv("DB_PATH")
if DB_PATH is None:
    raise FileNotFoundError("DB_PATH must be specified to access database")

EXT_DB_PATH = os.getenv("EXT_DB_PATH")
if EXT_DB_PATH is None:
    raise FileNotFoundError("EXT_DB_PATH must be specified to save new data")

QUERY = os.getenv("QUERY")
if QUERY is None:
    raise ValueError("QUERY must be specified for program to run")

SEED = os.getenv("SEED")
if SEED is None:
    raise ValueError("SEED must be specified for reproducibility")

EMAIL = os.getenv("EMAIL")
if EMAIL is None:
    raise ValueError("EMAIL must be specified for NCBI tool calling")

Entrez.email = EMAIL

EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"

MODEL_NAME = os.getenv("MODEL_NAME")
if MODEL_NAME is None:
    raise ValueError("MODEL_NAME must be specified - either proprietary or local"
    )

MODEL_TYPE = os.getenv("MODEL_TYPE")
if MODEL_TYPE is None:
    raise ValueError("MODEL_TYPE must be specified")

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

if int(MODEL_TYPE) == 0:
    GENERATIVE_MODEL = dspy.LM(
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
elif int(MODEL_TYPE) == 1:
    API_KEY = os.getenv("API_KEY")
    if API_KEY is None:
        raise ValueError("Valid API_KEY must be specified to use the proprietary model")
    GENERATIVE_MODEL = dspy.LM(
        MODEL_NAME,
        api_key=API_KEY,
        max_tokens=10_000,
        temperature=0,
        verbose=False,
    )
else:
    raise ValueError("MODEL_TYPE must be 0 or 1")


dspy.configure(lm=GENERATIVE_MODEL)



async def main(query: str) -> str:
    agent = GNAgent(
        corpus_path=CORPUS_PATH,
        pcorpus_path=PCORPUS_PATH,
        db_path=DB_PATH,
        ext_db_path=EXT_DB_PATH,
        naturalize_prompt=naturalize_prompt,
        rephrase_prompt=rephrase_prompt,
        analyze_prompt=analyze_prompt,
        check_prompt=check_prompt,
        summarize_prompt=summarize_prompt,
        synthesize_prompt=synthesize_prompt,
        split_prompt=split_prompt,
        finalize_prompt=finalize_prompt,
        sup_prompt1=sup_prompt1,
        sup_prompt2=sup_prompt2,
        plan_prompt=plan_prompt,
        refl_prompt=refl_prompt,
        expert_prompt=expert_prompt,
    )
    output = await agent.handler(query)
    logging.info(f"\n\nSystem feedback: {output}")

    return output


if __name__ == "__main__":
    logging.basicConfig(
        filename="log_agent.txt",
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s %(message)s",
    )
    asyncio.run(main(QUERY))
