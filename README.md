<div align="center">
	<h1>GSurgeon: the genomic surgeon</h1>
</div>

## What is GSurgeon?

**GSurgeon** is an LLM tool to dissect biology of model organisms through genomic information. It leverages LLM capabilities to send dynamic requests in natural language to genomic databases and extract any biological information.

## What questions can you ask GSurgeon?

**GSurgeon** has been tested on questions related to model organisms involving markers, genes and traits. A few examples with `anthropic/claude-opus-4-8`...

1. Which genes on chromosome 1 of the mouse genome are related to inflammation and diabetes?
```
Using the NCBI Gene database (Mus musculus, chromosome 1) filtered for both " inflammation" and "diabetes," I identified **57 genes** on mouse chromosome 1 that carry functional annotations relevant to both processes. All were confirmed to physically map to chromosome 1 (reference assembly GRCm39 / NC_000067 .7) and to have inflammatory and/or metabolic/diabetes-related functions. Below is the compiled, annotated report, grouped by functional theme with genomic positions.

## Genes with the strongest dual (inflammation + diabetes) evidence

These genes are explicitly implicated in diabetes mellitus (type 1 or type 2 AND have clear inflammatory/immune roles:
- **Irs1** (insulin receptor substrate 1; ~82.2 Mb): central node of insulin/IGF signaling; used to study type 2 diabetes mellitus; insulin resistance is intimately tied to chronic low-grade inflammation.
- **Il10** (interleukin 10; ~130.9 Mb): key anti-inflammatory/immunosuppressive cytokine; strongly studied in autoimmune diabetes and metabolic inflammation.
- **Ctla4** (cytotoxic T-lymphocyte-associated protein 4; ~60.9 Mb): negative regulator of T-cell responses; human ortholog implicated in **type 1 diabetes mellitus**.
- **Cd28** (CD28 antigen; ~60.8 Mb): T-cell costimulation; human ortholog implicated in **type 1 diabetes mellitus** and multiple sclerosis.
- **Icos** (inducible T-cell costimulator; ~61.0 Mb): T-cell costimulation/tolerance; autoimmune-associated.
- **Il1r1** (interleukin 1 receptor type I; ~40.3 Mb): IL-1 signaling; used to study **type 1 diabetes mellitus**; IL-1β drives beta-cell inflammation.
- **Fasl / Faslg** (Fas ligand; ~161.6 Mb): extrinsic apoptosis; used to study **type 1 diabetes mellitus** (beta-cell death).
- **Tnfsf4** (OX40 ligand; ~161.2 Mb): T-cell activation/inflammatory response; used to study **type 1 diabetes mellitus**; also "atherosclerosis 1."
- **Sell** (L-selectin/CD62L; ~163.9 Mb): leukocyte adhesion/homing; used to study **type 1 diabetes mellitus**.
- **Pdcd1** (PD-1; ~93.97 Mb): immune-checkpoint, negative regulator of immune response; autoimmune disease-associated.
- **Hsd11b1** (11β-hydroxysteroid dehydrogenase 1; ~192.9 Mb): regenerates active glucocorticoids; human ortholog implicated in **type 2 diabetes mellitus**, obesity, hypertension; glucocorticoids modulate inflammation.
- **Adipor1** (adiponectin receptor 1; ~134.3 Mb): mediates adiponectin's insulin-sensitizing and anti-inflammatory effects; knockout mice show impaired
...
```

2. List traits measured in GeneNetwork that are related to diabetes.

```
GeneNetwork contains a broad range of diabetes-related traits, which fall into several distinct categories. Below is a curated list of representative traits identified in the GeneNetwork RDF knowledge base.

## 1. Classical (BXD) Phenotype Traits Measuring Diabetes Physiology

These are directly measured phenotypes, mostly from BXD mouse populations using chemically induced diabetes models (streptozotocin/STZ or alloxan):
**Streptozotocin (STZ)-induced type I diabetes — kidney/glomerular traits:**
- **BXD_18404** – Urinary albumin/creatinine ratio (ACR, µg/mg) following long-term STZ-induced diabetes
- **BXD_18405** – Podocytes (WT1-positive cells) per glomerular section following long-term STZ diabetes
- **BXD_18406 / BXD_18410** – Wilms tumor 1 (WT1)-negative cells per glomerular section/area
- **BXD_18407** – Glomerular sectional area (GSA) following long-term STZ diabetes
- **BXD_18408** – Total cell density in glomerular sections following STZ diabetes
- **BXD_18409** – Podocyte cell density (WT1-positive per 1000 µm²) following STZ diabetes
- **BXD_12592–12600** – A series of kidney morphology traits (podocyte counts, total glomerular cell counts, glomerular surface area, WT1-positive/negative cell counts) following long-term STZ diabetes, including measurements 6 months after type I diabetes induction
- **BXD_17551 / BXD_17552** – Diabetes model, STZ-induced albumin-to-creatinine ratio (and log ratio) in males at 10–12 weeks of age

**Alloxan-induced diabetes traits:**
- **BXD_15958** – Alloxan-induced diabetes (80 mg/kg iv, killing pancreatic beta cells), retinopathy severity score (ordinal scale)
- **BXD_17020 / BXD_17021** – Kidney weight (as % body weight) at 7 and 21 days after alloxan treatment
- **BXD_17023** – Eye weight (as % body weight) at 7 days after alloxan treatment
...
```

Queries in the realm of biology and genomics are all supported.

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
