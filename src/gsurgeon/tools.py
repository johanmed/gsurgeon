# Specialized modules for researcher


class Naturalize(dspy.Signature):
    text: list[BaseMessage] = dspy.InputField()
    answer: str = dspy.OutputField(desc="Natural English sentence")


naturalize_pred = dspy.Predict(Naturalize)


class Rephrase(dspy.Signature):
    input_text: list[BaseMessage] = dspy.InputField()
    existing_history: list[BaseMessage] = dspy.InputField()
    answer: str = dspy.OutputField(desc="Reformulated query")


rephrase_pred = dspy.Predict(Rephrase)


class Analyze(dspy.Signature):
    context: list[BaseMessage] = dspy.InputField()
    existing_history: list[BaseMessage] = dspy.InputField()
    input_text: list[BaseMessage] = dspy.InputField()
    answer: str = dspy.OutputField(desc="Analysis (≤200 words)")


analyze_pred = dspy.Predict(Analyze)


class Check(dspy.Signature):
    answer: list[BaseMessage] = dspy.InputField()
    input_text: list[BaseMessage] = dspy.InputField()
    decision: str = dspy.OutputField(desc='"yes" or "no"')


check_pred = dspy.Predict(Check)


class Summarize(dspy.Signature):
    full_context: list[BaseMessage] = dspy.InputField()
    summary: str = dspy.OutputField(desc="Bullet-point summary")


summarize_pred = dspy.Predict(Summarize)


class Synthesize(dspy.Signature):
    input_text: list[BaseMessage] = dspy.InputField()
    updated_history: list[BaseMessage] = dspy.InputField()
    conclusion: str = dspy.OutputField(desc="Final paragraph")


synthesize_pred = dspy.Predict(Synthesize)


class Subquery(dspy.Signature):
    query: list[BaseMessage] = dspy.InputField()
    answer: list[str] = dspy.OutputField(desc="The list of smaller tasks")


subquery = dspy.Predict(Subquery)


class Finalize(dspy.Signature):
    query: list[BaseMessage] = dspy.InputField()
    subqueries: list[BaseMessage] = dspy.InputField()
    answers: list[BaseMessage] = dspy.InputField()
    conclusion: str = dspy.OutputField(desc="Final answer")


finalize_pred = dspy.Predict(Finalize)


# Specialized ReAct architecture for expert


def search_ncbi(database: str, term: str, max_results: int = 10) -> Any:
    handle = esearch(db=database, term=term, retmax=max_results)
    records = read(handle)
    handle.close()
    return records


search_ncbi = dspy.Tool(
    name="search_ncbi",
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


fetch_record = dspy.Tool(
    name="fetch_record",
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


summarize_record = dspy.Tool(
    name="summarize_record",
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
    solution: str = dspy.OutputField(desc="The answer to the query")


class React(dspy.Module):
    def __init__(self):
        super().__init__()
        self.tools = [search_ncbi, fetch_record, summarize_record]

        self.react = dspy.ReAct(
            signature=ReactSig,
            tools=self.tools,
            max_iters=50,  # maximum number of steps for reasoning and tool calling
        )

    def forward(self, query: list[BaseMessage]):
        return self.react(query=query)


class Plan(dspy.Signature):
    background: list[BaseMessage] = dspy.InputField()
    answer: str = dspy.OutputField(desc="The plan to solve the task")
    reasoning: str = dspy.OutputField(
        desc="Concise explanation of the output in 50 words"
    )


# Module to make plan
plan = dspy.Predict(Plan)


class Tune(dspy.Signature):
    background: list[BaseMessage] = dspy.InputField()
    answer: str = dspy.OutputField(desc="The new questions")
    reasoning: str = dspy.OutputField(
        desc="Concise explanation of the output in 50 words"
    )


# Module to tune reflection
tune = dspy.Predict(Tune)


class Decide(dspy.Signature):
    background: list[BaseMessage] = dspy.InputField()
    next_decision: Literal["researcher", "reflector", "expert", "end"] = (
        dspy.OutputField(desc="The next step to take based on instructions")
    )
    reasoning: str = dspy.OutputField(
        desc="Concise explanation of the decision in 50 words"
    )


# Module to manage system
supervise = dspy.Predict(Decide)


class End(dspy.Signature):
    messages: list[BaseMessage] = dspy.InputField()
    feedback: str = dspy.OutputField(
        desc="Detailed and comprehensive final feedback combining AI outputs in the list of messages and linking them when necessary"
    )


# Module to wrap up
end = dspy.Predict(End)

class AgentState(BaseModel):
    """
    Represents agent state
    Avails 02 attributes to allow communication between agents
    """

    messages: Annotated[list[BaseMessage], add_messages]
    next_decision: Literal["researcher", "planner", "reflector", "expert", "end"]


class ResearcherState(TypedDict):
    """
    Represents state of the agent researcher
    Avails 05 attributes to allow communication between its subcomponents
    """

    input_text: str
    chat_history: list[str]
    context: list[str]
    answer: str
    should_continue: str
