"""Modules with tools for multi-agent system"""

import asyncio
import concurrent.futures
import json
from typing import Any

import dspy
import httpx
from Bio.Entrez import efetch, esearch, esummary, read
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel
from SPARQLWrapper import JSON, SPARQLWrapper
from typing_extensions import Annotated, Literal


class Split(dspy.Signature):
    """Split query into multiple atomic subqueries easier to handle for better satisfaction"""

    query: str = dspy.InputField()
    answer: list[str] = dspy.OutputField(desc="The list of smaller tasks")


def split_query(query: str) -> list[str]:
    split = dspy.Predict(Split)
    return split(query=query).get("answer")


splitter = dspy.Tool(
    name="splitter",
    desc="Process a query by splitting into atomic subqueries for efficiency",
    args={
        "query": {
            "type": "string",
            "desc": "Query to process",
        },
    },
    func=split_query,
)


class Check(dspy.Signature):
    """Check if info is relevant to query"""

    query: str = dspy.InputField()
    info: str = dspy.InputField()
    decision: str = dspy.OutputField(desc="Say 'yes' or 'no'")


def check_relevance(query: str, info: str) -> str:
    check = dspy.Predict(Check)
    return check(query=query, info=info).get("decision")


checker = dspy.Tool(
    name="checker",
    desc="Check if information previously extracted is relevant for the query",
    args={
        "query": {
            "type": "string",
            "desc": "Query to address",
        },
        "info": {
            "type": "string",
            "desc": "Information extracted in attempt to provide answer to query",
        },
    },
    func=check_relevance,
)


class Rephrase(dspy.Signature):
    """Reformulate query given target and context accumulated so far"""

    query: str = dspy.InputField()
    target: str = dspy.InputField()
    background: str = dspy.InputField()
    reformulation: str = dspy.OutputField(desc="Reformulated query")


def rephrase_query(query: str, target: str, background: str) -> str:
    rephrase = dspy.Predict(Rephrase)
    return rephrase(query=query, target=target, background=background).get(
        "reformulation"
    )


reformulator = dspy.Tool(
    name="reformulator",
    desc="Reformulate the query to be next processed in light of the context accumulated so far (background) and the target",
    args={
        "query": {
            "type": "string",
            "desc": "Query to be reformulated",
        },
        "target": {
            "type": "string",
            "desc": "Original query or target",
        },
        "background": {
            "type": "string",
            "desc": "Accumulated context in effort to achieve the target",
        },
    },
    func=rephrase_query,
)


def search_ncbi(database: str, term: str, max_results: int = 10) -> str:
    handle = esearch(db=database, term=term, retmax=max_results)
    records = read(handle)
    handle.close()
    # Order records for determinism
    if isinstance(records, dict) and "IdList" in records:
        records["IdList"] = sorted(records["IdList"])
    return json.dumps(records, sort_keys=True)


ncbi_searcher = dspy.Tool(
    name="ncbi_searcher",
    desc="Search an NCBI database (e.g., nucleotide, protein, pubmed) for a term",
    args={
        "database": {
            "type": "string",
            "desc": "Database name like 'nucleotide' or 'pubmed'",
        },
        "term": {"type": "string", "desc": "Search term or query"},
        "max_results": {
            "type": "integer",
            "desc": "Max results (default 200)",
            "default": 200,
        },
    },
    func=search_ncbi,
)


def fetch_record(database: str, record_id: str, rettype: str) -> str:
    handle = efetch(db=database, id=record_id, rettype=rettype, retmode="text")
    result = handle.readline().strip()
    handle.close()
    return result


record_fetcher = dspy.Tool(
    name="record_fetcher",
    desc="Fetch a record from an NCBI database (e.g., nucleotide, protein, pubmed)",
    args={
        "database": {
            "type": "string",
            "desc": "Database name like 'nucleotide' or 'pubmed'",
        },
        "record_id": {"type": "string", "desc": "Identifier of record"},
        "rettype": {"type": "string", "desc": "Return type compatible with database"},
    },
    func=fetch_record,
)


def summarize_record(database: str, record_id: str) -> str:
    handle = esummary(db=database, id=record_id)
    result = read(handle)
    handle.close()
    # If a list of summaries, sort by Id for determinism
    if isinstance(result, list):
        result = sorted(result, key=lambda x: x.get("Id", ""))
    return json.dumps(result, sort_keys=True)


record_synthesizer = dspy.Tool(
    name="record_synthesiser",
    desc="Get summary on a record from an NCBI database (e.g., nucleotide, protein, pubmed)",
    args={
        "database": {
            "type": "string",
            "desc": "Database name like 'nucleotide' or 'pubmed'",
        },
        "record_id": {"type": "string", "desc": "Identifier of record"},
    },
    func=summarize_record,
)


class QueryTranslation(dspy.Signature):
    """Compare object snapshot in schema hint to keywords in the original query to find best semantic matches.
    Use matches to generate valid SPARQL SELECT queries that can retrieve relevant information for the query.
    CRITICAL:
    1. Every query MUST start with the PREFIX declarations. Only use declared prefixes.
    2. Leverage as many schema hints as possible.

    When querying SPARQL, prefer fast, efficient SPARQL SELECT queries
    that avoid Virtuoso timeouts (504 errors).

    PERFORMANCE RULES:
    1. Always add `LIMIT` - start with `LIMIT 50`, increase only if needed. Never omit `LIMIT`.
    2. Never use `SELECT *` - list only the variables you actually need.
    3. Avoid expensive operations: no Cartesian products, no cross joins, no full graph scans.
    4. Use specific FILTER patterns that leverage indexes:
    - Prefer `STRSTARTS(?label, "prefix")` over `CONTAINS` or regex.
    - Avoid `FILTER regex(...)` - it disables indexes.
    - Use `FILTER(?value = "exact")` or `IN` with small lists.
    5. Prefer property paths over multiple joins when traversing a chain.
    6. Use VALUES blocks for small sets of constants instead of UNION or OPTIONAL.
    7. Avoid ORDER BY on large result sets - if needed, combine with `LIMIT` and a narrow `WHERE` clause.
    8. Never use nested subqueries unless absolutely necessary; flatten them.
    9. Use `OPTIONAL` only for truly optional patterns – otherwise, use a simple triple pattern.
    """

    original_query: str = dspy.InputField(desc="User query")
    schema_hint: str = dspy.InputField(desc="GeneNetwork schema from Virtuoso")
    translated_queries: list[str] = dspy.OutputField(
        desc="Top 10 valid SPARQL SELECT query with PREFIX declarations."
    )


def make_sparql_tool(sparql_uri: str) -> dspy.Tool:

    def sparql_fetcher(query: str) -> Any:

        def build_schema_hint(sparql_uri: str) -> str:
            """Build a compact schema hint from the live Virtuoso endpoint."""
            _PREFIX_MAP = {
                "http://rdf.genenetwork.org/v1/term/": "gnt",
                "http://rdf.genenetwork.org/v1/category/": "gnc",
                "http://rdf.genenetwork.org/v1/id/": "gn",
                "http://purl.org/dc/terms/": "dct",
                "http://www.w3.org/ns/dcat#": "dcat",
                "http://www.w3.org/2000/01/rdf-schema#": "rdfs",
                "http://www.w3.org/2004/02/skos/core#": "skos",
                "http://www.w3.org/1999/02/22-rdf-syntax-ns#": "rdf",
                "http://www.w3.org/2002/07/owl#": "owl",
                "http://purl.org/linked-data/cube#": "qb",
                "http://purl.org/linked-data/sdmx/2009/measure#": "sdmx-measure",
                "http://rdf-vocabulary.ddialliance.org/xkos#": "xkos",
                "https://schema.org/": "schema",
                "http://rdf.ncbi.nlm.nih.gov/pubmed/": "pubmed",
                "http://xmlns.com/foaf/0.1/": "foaf",
                "http://purl.org/spar/fabio/": "fabio",
                "http://prismstandard.org/namespaces/basic/2.0/": "prism",
            }

            def uri_to_qname(uri: str) -> str:
                """Convert a full URI to a prefixed name, or return the URI in angle brackets."""
                for ns, prefix in sorted(_PREFIX_MAP.items(), key=lambda x: -len(x[0])):
                    if uri.startswith(ns):
                        return f"{prefix}:{uri[len(ns):]}"
                return f"<{uri}>"

            def fetch_schema(sparql_uri: str) -> tuple[set[str], set[str], set[str]]:
                """Fetch literal and object properties from the live Virtuoso endpoint.
                Return (literal_props, iri_props) where each is a set of full URIs.
                """
                sparql = SPARQLWrapper(sparql_uri)
                sparql.setReturnFormat(JSON)

                literal_query = """
                SELECT DISTINCT ?p
                WHERE { ?s ?p ?o . FILTER isLiteral(?o) }
                """
                sparql.setQuery(literal_query)
                lit_result = sparql.queryAndConvert()
                literal_props = {
                    b["p"]["value"]
                    for b in lit_result.get("results", {}).get("bindings", [])
                    if b.get("p")
                }

                iri_query = """
                SELECT DISTINCT ?p
                WHERE { ?s ?p ?o . FILTER isIRI(?o) }
                """
                sparql.setQuery(iri_query)
                iri_result = sparql.queryAndConvert()
                iri_props = {
                    b["p"]["value"]
                    for b in iri_result.get("results", {}).get("bindings", [])
                    if b.get("p")
                }

                return literal_props, iri_props

            literal_props, iri_props = fetch_schema(sparql_uri)
            return f"""=== GENENETWORK SCHEMA (from Virtuoso) ===
            PREFIX dcat: <http://www.w3.org/ns/dcat#>
            PREFIX gn: <http://rdf.genenetwork.org/v1/id/>
            PREFIX dct: <http://purl.org/dc/terms/>
            PREFIX gnc: <http://rdf.genenetwork.org/v1/category/>
            PREFIX gnt: <http://rdf.genenetwork.org/v1/term/>
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
            PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

            LITERAL PROPERTIES (object is a string/number/date):
            {" ,".join([uri_to_qname(uri) for uri in literal_props])}

            IRI PROPERTIES (object is a URI / another resource):
            {" ,".join([uri_to_qname(uri) for uri in iri_props])}

            SPECIAL HINTS FOR TRIPLE GENERATION:
            1. To check if a trait is mapped, use: `?trait a gnt:mappedTrait .`
            2. To get trait id, use: `?trait gnt:traitId ?trait_id .`
            3. To fetch trait description, use: `?trait dct:description ?trait_description .`
            4. To extract lod score at a specific locus, use: `?trait gnt:locus ?locus; gnt:lodScore ?lod_score .`
            5. To fetch information related to QTL for a trait, use:
            `?trait gnt:qtlChr ?chromosome; gnt:qtlStart ?start; gnt:qtlStop ?stop; gnt:qtlLOD ?lod_score .`

            CRITICAL RULES:
            1. Only use properties listed above. Do NOT invent new ones.
            2. Literal properties give strings/numbers — use FILTER, not ?o a ...
            3. Object properties link to other resources — you can chain ?o a <Class>.
            4. Do NOT use taxon: for species. Use gn:Mus_musculus, gn:Rattus_norvegicus, gn:Homo_sapiens, etc.
            5. gnt:has_trait_page gives the URL directly. Never build trait URLs manually.
            """

        schema_hint = build_schema_hint(sparql_uri)
        translate_sparql = dspy.Predict(QueryTranslation)
        sparql_queries = translate_sparql(
            original_query=query, schema_hint=schema_hint
        ).get("translated_queries")

        async def run_sparql(
            sparql_uri: str,
            query: str,
            max_retries: int = 3,
            base_delay: float = 2,
        ) -> dict:
            """Execute a single SPARQL query with retry + exponential jitter via httpx."""
            client = httpx.AsyncClient(timeout=5000)
            for attempt in range(max_retries):
                try:
                    resp = await client.post(
                        sparql_uri,
                        data={"query": query},
                        headers={"Accept": "application/sparql-results+json"},
                    )
                    resp.raise_for_status()
                    return resp.json()
                except httpx.HTTPStatusError as e:
                    if (
                        e.response.status_code in (504, 503, 502)
                        and attempt < max_retries - 1
                    ):
                        await asyncio.sleep(
                            base_delay * (2**attempt) + random.uniform(0, 1)
                        )
                        continue
                    raise
            return {}

        async def sparql_fetch(
            sparql_queries: list[str],
            sparql_uri: str,
            max_retries: int = 3,
            base_delay: float = 0.5,
        ) -> str:
            """Execute *sparql_queries* concurrently against *sparql_uri*."""
            if not sparql_queries:
                return "No SPARQL queries to run."

            async def _fetch_one(query: str, idx: int) -> str:
                try:
                    result = await run_sparql(
                        sparql_uri, query, max_retries, base_delay
                    )
                    bindings = result.get("results", {}).get("bindings", [])
                    return f"Query {idx} succeeded ({len(bindings)} rows): {bindings}"
                except Exception as e:
                    return f"Query {idx} failed: {e}\nQuery was:\n{query}"

            tasks = [_fetch_one(q, i) for i, q in enumerate(sparql_queries)]
            results = await asyncio.gather(*tasks)
            return "\n\n".join(results)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future = executor.submit(
                asyncio.run, sparql_fetch(sparql_queries, sparql_uri)
            )
            return future.result()

    return dspy.Tool(
        name="sparql_fetcher",
        desc="Fetch RDF data around GeneNetwork data through SPARQL",
        args={
            "query": {
                "type": "string",
                "desc": "SPARQL query to run to fetch relevant data",
            },
        },
        func=sparql_fetcher,
    )


class ReactSig(dspy.Signature):
    query: list[BaseMessage] = dspy.InputField()
    solution: str = dspy.OutputField(desc="The final answer to the query")


class Research(dspy.Module):
    """
    Address a query or plan to completion using GeneNetwork resources only.
    For effficiency, only call a tool when it is strictly necessary in completing the next task.
    Use splitter when input query is too complex to be handled in a single step.
    Harness the reformulator to clarify a request when it seems ambiguous.
    To get a specific information, call the fetcher. It has access to data and can extract any information.
    Once an information is extracted, check its relevance with the checker before proceeding.
    """

    def __init__(self):
        super().__init__()
        fetcher = make_sparql_tool("https://sparql.genenetwork.org/sparql")
        self.tools = [splitter, checker, reformulator, fetcher]

        self.react = dspy.ReAct(
            signature=ReactSig,
            tools=self.tools,
            max_iters=10,  # maximum number of steps for reasoning and tool calling
        )

    def forward(self, query: list[BaseMessage]):
        return self.react(query=query)


class Consult(dspy.Module):
    """
    Address a query or plan to completion using NCBI resources only.
    For effficiency, only call a tool when it is strictly necessary in completing the next task.
    Use splitter when input query is too complex to be handled in a single step.
    Harness the reformulator to clarify a request when it seems ambiguous.
    Extract answers from NCBI by performing first a search with ncbi_searcher.
    When search results contain records, fetch information with record_fetcher.
    For records with a lot of data specifically, take some time to synthesize informations.
    Check relevance of generated information with the checker before proceeding.
    """

    def __init__(self):
        super().__init__()
        self.tools = [
            splitter,
            checker,
            reformulator,
            ncbi_searcher,
            record_fetcher,
            record_synthesizer,
        ]

        self.react = dspy.ReAct(
            signature=ReactSig,
            tools=self.tools,
            max_iters=10,
        )

    def forward(self, query: list[BaseMessage]):
        return self.react(query=query)


class Plan(dspy.Signature):
    """Generate plan to solve query in background"""

    background: list[BaseMessage] = dspy.InputField()
    answer: str = dspy.OutputField(desc="The plan to solve the task")
    reasoning: str = dspy.OutputField(
        desc="Concise explanation of the output in 50 words"
    )


class Tune(dspy.Signature):
    """Make recommendations to improve user satisfaction to answer generated so far to query"""

    background: list[BaseMessage] = dspy.InputField()
    answer: str = dspy.OutputField(desc="The new questions")
    reasoning: str = dspy.OutputField(
        desc="Concise explanation of the output in 50 words"
    )


class Supervise(dspy.Signature):
    """
    Decide the next action the system should take.
    To select the next step, you must take into account the query and the curent context.
    If the query is not related to GeneNetwork traits, do not call gn_researcher. ncbi_expert should be the main actor.
    Similarly, do not call the ncbi_expert if the query is GeneNetwork specific.
    Call the reflector only to improve generation from gn_researcher and ncbi_expert.
    Act on suggestions proposed by reflector using the most appropriate actor between gn_researcher and ncbi_expert depending on the query.
    End execution if there is nothing else to do.
    """

    background: list[BaseMessage] = dspy.InputField()
    next_decision: Literal["gn_researcher", "ncbi_expert", "reflector", "end"] = (
        dspy.OutputField(desc="The next step to take based on instructions")
    )
    reasoning: str = dspy.OutputField(
        desc="Concise explanation of the decision in 50 words"
    )


class Finalize(dspy.Signature):
    """Build the final synthesis to send back to the user in less than 200 words"""

    messages: list[BaseMessage] = dspy.InputField()
    feedback: str = dspy.OutputField(
        desc="Detailed and comprehensive final feedback combining AI outputs in the list of messages and linking them when necessary"
    )


class AgentState(BaseModel):
    """
    Represent agent state
    Avail 02 attributes to allow communication between agents
    """

    messages: Annotated[list[BaseMessage], add_messages]
    next_decision: Literal[
        "gn_researcher", "planner", "reflector", "ncbi_expert", "end"
    ]
