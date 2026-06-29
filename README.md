# GSurgeon: the genomic surgeon

## What is GSurgeon?

**GSurgeon** is an AI tool to dissect biology of model organisms through genomic information. It leverages LLM capabilities to send dynamic requests in natural language to genomic databases and extract any biological information.

## What questions can you ask GSurgeon?

**GSurgeon** has been tested on questions related to model organisms involving markers, genes and traits.

As such it has good performance on queries such as:

- Which genes on chromosome 1 of the mouse genome are related to inflammation and diabetes at the same time?
- List traits measured in GeneNetwork that are related to diabetes.

Other queries in the realm of biology and genomics are also possible.

## How to install and run GSurgeon?

#### 1. Get the source code

You can clone this repository.

```bash
git clone https://github.com/johanmed/gsurgeon.git
```

#### 2. Install dependencies

You can install dependencies for this project using:

```
poetry install
```

Poetry needs to already be installed in your machine for the command above to run. See [Poetry installation guide](https://python-poetry.org/docs/).

Alternatively, you can create a Python virtual environment and install the dependencies in `pyproject.toml` one by one in it.

```
python3 -m venv .venv
source .venv/bin/activate
pip install "langchain-core>=1.4.0,<2.0.0"
pip install "langgraph>=1.2.0,<2.0.0"
...
```

This is a fallback in case you experience challenges getting started with poetry. It takes more work, so we recommend using poetry which manages everything automatically.

#### 3. Set tool parameters

**GSurgeon** expects a number of parameters to be defined for the surgery:

- N_ITERATIONS: number of operations
- MODEL_NAME: DSPy model identifier
- API_KEY: provider key
- EMAIL: email address for NCBI authentication

We recommend creating them in an environment file. For more details, see file `env_example`.

#### 4. Add gsurgeon path to your search path

```bash
export PATH="$PATH:/path/to/project"
```

Replace the path above by yours. You can also add it to your file `~/.bashrc`.

#### 5. Run your query

```bash
gsurgeon --env-file env_example "Which genes on chromosome 1 of the mouse genome are related to inflammation and diabetes at the same time?"
```

Replace the query above by yours.

## Why use GSurgeon?

#### 1. Access to up-to-date biological information

Accessing genomic information is a pain. It requires knowledge of right databases to query but also skills to dig deep and find relevant information. **GSurgeon** makes the process easier for the community by providing a simpler, yet powerful interface to interact in real time with biological databases.

In the research ecosystem, this can be used for a variety of applications:

- literature review
- cross-checking of research findings against current knowledge
- hypothesis exploration
- biological link discovery
- advanced bioinformatic analyses

#### 2. Prevent hallucination, trust a bit more language models used in biology

Despite advances in language AI, hallucination remains a serious concern in biological research. **GSurgeon** offers a scalable solution by grounding generation in true information from biological databases. Current databases supported include:
- [GeneNetwork](https://genenetwork.org/): database service to explore biology of model organisms with bioinformatic tools
- [NCBI](https://www.ncbi.nlm.nih.gov/): database service for access and analysis of biological information

#### 3. Empower your LLM to handle with surgical precision the hard work for you with no limits

**GSurgeon** exploits reasoning capabilities of LLM to orchestrate the search of biological information. Using its knowledge of biological databases, it finds dynamically the best approach of answering or completing the task you have in mind.

The execution logic is abstracted to give full control to the agents. No need for extra coding!

The tool footprint is lightweight. Most of the computational resources required to run the system are handled by the provider. No need to have monstruous specs to get started!

**GSurgeon** can be executed on the command-line on any model, provided sufficient training.
