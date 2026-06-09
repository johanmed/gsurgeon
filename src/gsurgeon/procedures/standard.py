"""Module with standard procedures (dspy modules) for GSurgeon"""

import dspy
from langchain_core.messages import BaseMessage
from typing_extensions import Literal


class Plan(dspy.Signature):
    """
    Generate plan to solve query in background.
    For reproducibility, seed the plan generation on the following master thought:
    The query submitted by the user can be addressed in less than 5 straightforward and targeted steps. No need to complicate things.
    """

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
    When the query is related to genes, finemapping or network analysis, you must call the ncbi_expert and not the gn_researcher.
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
    """Build the final synthesis to send back to the user in less than 500 words"""

    messages: list[BaseMessage] = dspy.InputField()
    feedback: str = dspy.OutputField(
        desc="Detailed and comprehensive final feedback combining AI outputs in the list of messages and linking them when necessary"
    )


class Reproduce(dspy.Signature):
    """
    Extract answers that are consistent across results.
    Do not include information that is missing in some results for reproducibility.
    Synthesize a coherent and detailed solution to the query using answer consensus.
    """

    query: str = dspy.InputField()
    results: list = dspy.InputField(
        desc="List of results generated to the same query by the system"
    )
    consensus: str = dspy.OutputField(
        desc="Final output built from consistent answers across results"
    )


class Resample(dspy.Signature):
    """
    Examine the query and reformulate it in a new way.
    The reformulation should be a resampling with replacement of the original query and must have a similar meaning.
    Feel free to make any modification you want on the instructions. The more you tweak the instructions, the better it is.
    Use as many synonyms and paraphrasing as possible.
    """

    query: str = dspy.InputField()
    reformulation: str = dspy.OutputField(
        desc="Reformulated query"
    )
