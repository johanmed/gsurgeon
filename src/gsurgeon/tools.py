"""Modules with tools for multi-agent system"""

from Bio import Entrez
from Bio.Entrez import efetch, esearch, esummary, read
from langchain_core.messages import BaseMessage


class Split(dspy.Signature):
    query: str = dspy.InputField()
    answer: list[str] = dspy.OutputField(desc="The list of smaller tasks")

def split_query(query: str) -> list[str]:
    """Split query into multiple atomic subqueries easier to handle for better satisfaction"""
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
    return rephrase(query=query, target=target, background=background).get("reformulation")

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

def search_ncbi(database: str, term: str, max_results: int = 10) -> Any:
    handle = esearch(db=database, term=term, retmax=max_results)
    records = read(handle)
    handle.close()
    return records

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
            "desc": "Max results (default 10)",
            "default": 10,
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


def summarize_record(database: str, record_id: str) -> Any:
    handle = esummary(db=database, id=record_id)
    result = read(handle)
    handle.close()
    return result

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


class ReactSig(dspy.Signature):
    query: list[BaseMessage] = dspy.InputField()
    solution: str = dspy.OutputField(desc="The final answer to the query")


class Research(dspy.Module):
    def __init__(self):
        super().__init__()
        self.tools = [splitter, checker, reformulator]

        self.react = dspy.ReAct(
            signature=ReactSig,
            tools=self.tools,
            max_iters=50,  # maximum number of steps for reasoning and tool calling
        )

    def forward(self, query: list[BaseMessage]):
        return self.react(query=query)

class Consult(dspy.Module):
    def __init__(self):
        super().__init__()
        self.tools = [splitter, checker, reformulator, ncbi_searcher, record_fetcher, record_synthesizer]

        self.react = dspy.ReAct(
            signature=ReactSig,
            tools=self.tools,
            max_iters=50,
        )

    def forward(self, query: list[BaseMessage]):
        return self.react(query=query)

class Plan(dspy.Signature):
    background: list[BaseMessage] = dspy.InputField()
    answer: str = dspy.OutputField(desc="The plan to solve the task")
    reasoning: str = dspy.OutputField(
        desc="Concise explanation of the output in 50 words"
    )
plan = dspy.Predict(Plan)

class Tune(dspy.Signature):
    background: list[BaseMessage] = dspy.InputField()
    answer: str = dspy.OutputField(desc="The new questions")
    reasoning: str = dspy.OutputField(
        desc="Concise explanation of the output in 50 words"
    )
tune = dspy.Predict(Tune)

class Decide(dspy.Signature):
    background: list[BaseMessage] = dspy.InputField()
    next_decision: Literal["researcher", "reflector", "expert", "end"] = (
        dspy.OutputField(desc="The next step to take based on instructions")
    )
    reasoning: str = dspy.OutputField(
        desc="Concise explanation of the decision in 50 words"
    )
supervise = dspy.Predict(Decide)

class Finalize(dspy.Signature):
    messages: list[BaseMessage] = dspy.InputField()
    feedback: str = dspy.OutputField(
        desc="Detailed and comprehensive final feedback combining AI outputs in the list of messages and linking them when necessary"
    )
finalize = dspy.Predict(Finalize)

class AgentState(BaseModel):
    """
    Represents agent state
    Avails 02 attributes to allow communication between agents
    """

    messages: Annotated[list[BaseMessage], add_messages]
    next_decision: Literal["researcher", "planner", "reflector", "expert", "end"]
