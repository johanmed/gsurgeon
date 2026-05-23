"""Modules with general tools for GSurgeon"""

import dspy


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
