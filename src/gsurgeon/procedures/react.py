"""Module with ReAct procedures (dspy programs) for GSurgeon"""

import dspy
from gsurgeon.tools.general import checker, reformulator, splitter
from gsurgeon.tools.gn import make_sparql_tool
from gsurgeon.tools.ncbi import ncbi_searcher, record_fetcher, record_synthesizer
from langchain_core.messages import BaseMessage


class ReactSig(dspy.Signature):
    query: list[BaseMessage] = dspy.InputField()
    reasoning: str = dspy.OutputField(desc="Concise explanation of solution")
    solution: str = dspy.OutputField(
        desc="Final answer to the query in 2000 words max making use of all relevant information in accumulated context."
    )


class Research(dspy.Module):
    """
    Address a query or plan to completion using GeneNetwork resources only.
    For efficiency, only call a tool when it is strictly necessary in completing the next task.
    Use splitter when input query is too complex to be handled in a single step.
    Harness the reformulator to clarify a request when it seems ambiguous.
    To get a specific information, call the fetcher. It has access to data and can extract any information.
    Once an information is extracted, check its relevance with the checker before proceeding.
    For reproducibility, seed your reasoning on the following master thoughts:
        1. GeneNetwork has the data requested by the user and it can be obtained with targeted SPARQL queries.
        2. Reasoning does not need to be complicated. Prefer the simplest approach.
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
    Extract answers from NCBI by performing first a search with ncbi_searcher. The search terms must be as specific as possible.
    When search results contain records, fetch information with record_fetcher.
    For records with a lot of data specifically, take some time to synthesize informations.
    Check relevance of generated information with the checker before proceeding.
    Every information you extracted regarding genes and/or functions must be verified with another search using proper terms with ncbi_searcher.
    You must ascertain that all informations in the final answer are true at all cost.
    For reproducibility, seed your reasoning on the following master thoughts:
        1. NCBI has the data requested by the user and it can be obtained by using a combination of terms with AND or OR keywords.
        2. Reasoning does not need to be complicated. Prefer the simplest approach.
    For finemapping taks, always focus on the top 20 genes unless specified otherwise.
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
            max_iters=20,
        )

    def forward(self, query: list[BaseMessage]):
        return self.react(query=query)
