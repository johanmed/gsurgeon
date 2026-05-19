"""
Multi-agent system to dissect genomic information
Main module of the package
Author: Johannes Medagbe
Copyright (c) 2025
"""

import asyncio
import json
import logging
import os
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel
from typing_extensions import Annotated, TypedDict

from config import *
from prompts import *

warnings.filterwarnings("ignore")

@dataclass
class Search:
    """
    Represent Search Agent
    Input:
        max_iterations: maximum number of iterations allowed
    Operations:
        Initialization of multi-agent graph
        Run of query through system
    """
    max_iterations: int = 10
    resgraph: Any = field(init=False)

    def __post_init__(self):
        self.resgraph = self.initialize_resgraph()

    def split_query(self, query: str) -> list[str]:
        # Split query in researcher
        logging.info("Splitting query")

        split_prompt = [self.split_prompt, HumanMessage(query)]
        result = subquery(query=split_prompt)

        logging.info(f"Subqueries in split_query: {result}")
        result = result.get("answer")

        return result

    def analyze(self, state: ResearcherState) -> dict:
        """Addresses a query based on retrieved documents in researcher

        Args:
            state: researcher state

        Returns:
            researcher state updated with answer
        """

        logging.info("Analysing")

        context = (
            "\n".join(doc.page_content for doc in state.get("context", []))
            if state.get("context", [])
            else ""
        )

        truncated_context = str(context)[
            :25_000
        ]  # prehandle context length of large documents given model limit of 32_000

        existing_history = (
            "\n".join(state.get("chat_history", []))
            if state.get("chat_history", [])
            else ""
        )

        analyze_prompt = [self.analyze_prompt, HumanMessage(state["input_text"])]

        response = analyze_pred(
            input_text=analyze_prompt,
            context=[HumanMessage(truncated_context)],
            existing_history=[HumanMessage(existing_history)],
        )

        logging.info(f"Response in analyze: {response}")

        response = response.get("answer")
        should_continue = "check_relevance"

        return {
            "input_text": state["input_text"],
            "answer": response,
            "should_continue": should_continue,
            "context": state.get("context", []),
            "chat_history": state.get("chat_history", []),
        }

    def check_relevance(self, state: ResearcherState) -> dict:
        """Checks relevance of answer to query in researcher

        Args:
            state: researcher state

        Returns:
            researcher state updated with relevance status
        """

        logging.info("Checking relevance")

        answer = state["answer"]

        check_prompt = [self.check_prompt, HumanMessage(state["input_text"])]

        assessment = check_pred(input_text=check_prompt, answer=[HumanMessage(answer)])
        logging.info(f"Assessment in checking relevance: {assessment}")

        if assessment.get("decision") == "yes":
            should_continue = "summarize"
        else:
            should_continue = "end"
            answer = "Sorry, we are unable to \
                provide a valuable feedback due to lack of relevant data."

        return {
            "input_text": state["input_text"],
            "context": state.get("context", []),
            "answer": answer,
            "chat_history": state.get("chat_history", []),
            "should_continue": should_continue,
        }


    def initialize_resgraph(self) -> Any:
        graph_builder = StateGraph(ResearcherState)
        graph_builder.add_node("rephrase", self.rephrase)
        graph_builder.add_node("retrieve", self.retrieve)
        graph_builder.add_node("check_relevance", self.check_relevance)
        graph_builder.add_node("analyze", self.analyze)
        graph_builder.add_node("summarize", self.summarize)
        graph_builder.add_edge(START, "rephrase")
        graph_builder.add_edge("rephrase", "retrieve")
        graph_builder.add_edge("retrieve", "analyze")
        graph_builder.add_edge("analyze", "check_relevance")
        graph_builder.add_edge("summarize", END)
        graph_builder.add_conditional_edges(
            "check_relevance",
            lambda state: state.get("should_continue", "summarize"),
            {"summarize": "summarize", "end": END},
        )
        resgraph = graph_builder.compile(checkpointer=self.memory)

        return resgraph

    def researcher(self, state: AgentState) -> dict:
        """Researches a query

        Args:
            state: agent state containing query

        Returns:
            agent state updated with result
        """

        logging.info("Researching")
        if len(state.messages) < 3:  # handle first call to researcher
            input_text = state.messages[0]  # use original query
        else:
            input_text = state.messages[-1]  # use reflection insights
        input_text = input_text.content
        logging.info(f"Input in researcher: {input_text}")
        result = self.manage_subtasks(input_text)
        logging.info(f"Result in researcher: {result}")

        return {
            "messages": [result],
        }

    def expert(self, state: AgentState) -> dict:
        """Addresses a query using own model thinking and search tool through ReAct

        Args:
            state: agent state containing query

        Returns:
            agent state updated with answer
        """

        logging.info("Expert extracting knowledge")
        if len(state.messages) < 4:  # handle first call to expert
            input_text = state.messages[1] + state.messages[0] # use plan and query
        else:
            input_text = state.messages[-2]  # use reflection insights

        input_text = [self.expert_prompt, input_text]
        logging.info(f"Input in expert: {input_text}")

        react = React()
        result = react(query=input_text)

        logging.info(f"Result from expert: {result}")
        answer = result.get("solution")

        # Save information in database for reuse later by researcher
        metadata = {"source": f"New Document {self.ext_db._collection.count() + 1}"}
        self.ext_db.add_texts(
            texts=[answer],
            metadatas=[metadata],
        )
        self.ext_db.persist()

        return {
            "messages": [answer],
        }

    def planner(self, state: AgentState) -> dict:
        """Plans steps to tackle a problem

        Args:
            state: agent state specifying problem

        Returns:
            agent state updated with plan
        """

        logging.info("Planning")
        input_text = [self.plan_prompt] + state.messages
        logging.info(f"Input in planner: {input_text}")
        result = plan(background=input_text)
        logging.info(f"Result in planner: {result}")
        answer = result.get("answer")

        return {
            "messages": [answer],
        }

    def reflector(self, state: AgentState) -> dict:
        """Reflects about progress

        Args:
            state: agent state with current progress

        Returns:
            agent state updated with suggestions
        """

        logging.info("Reflecting")
        trans_map = {AIMessage: HumanMessage, HumanMessage: AIMessage}
        translated_messages = [self.refl_prompt, state.messages[0]] + [
            trans_map[msg.__class__](content=msg.content) for msg in state.messages[1:]
        ]
        logging.info(f"Input in reflector: {translated_messages}")
        result = tune(background=translated_messages)
        logging.info(f"Result in reflector: {result}")
        answer = result.get("answer")
        answer = (
            "Progress has been made. Use now all the resources to addess this new suggestion: "
            + answer
        )

        return {
            "messages": [HumanMessage(answer)],
        }

    def supervisor(self, state: AgentState) -> dict:
        """Manages interactions between other agents in system

        Args:
            state: agent state with relevant data

        Returns:
            agent state updated with next agent to call
        """

        logging.info("Supervising")
        messages = [
            ("system", self.sup_prompt1),
            *state.messages,
            ("system", self.sup_prompt2),
        ]

        if len(messages) > self.max_iterations:
            return {"next_decision": "end"}

        result = supervise(background=messages)
        logging.info(f"Result in supervisor: {result}")
        next_decision = result.get("next_decision")

        return {
            "next_decision": next_decision,
        }

    def initialize_globgraph(self) -> Any:
        graph_builder = StateGraph(AgentState)
        graph_builder.add_node("researcher", self.researcher)
        graph_builder.add_node("planner", self.planner)
        graph_builder.add_node("reflector", self.reflector)
        graph_builder.add_node("supervisor", self.supervisor)
        graph_builder.add_node("expert", self.expert)
        graph_builder.add_edge(START, "planner")
        graph_builder.add_edge("researcher", "supervisor")
        graph_builder.add_edge("expert", "supervisor")
        graph_builder.add_edge("planner", "researcher")
        graph_builder.add_edge("reflector", "researcher")
        graph_builder.add_conditional_edges(
            "supervisor",
            lambda state: state.next_decision,
            {
                "reflector": "reflector",
                "researcher": "researcher",
                "expert": "expert",
                "end": END,
            },
        )
        graph = graph_builder.compile()

        return graph

    async def invoke_globgraph(self, query: str) -> Any:
        graph = self.initialize_globgraph()
        initial_state = {
            "messages": [("human", query)],
            "next_decision": "planner",  # always plan first
        }
        result = await graph.ainvoke(initial_state)
        return result

    async def handler(self, query: str) -> str:
        """
        Main question handler of the system
        """
        global_result = await self.invoke_globgraph(query)
        end_prompt = global_result.get("messages")
        end_result = end(messages=end_prompt)
        end_result = end_result.get("feedback")

        first_result = global_result.get("messages")[
            2
        ].content  # get first researcher feedback
        second_result = global_result.get("messages")[
            3
        ].content  # get first expert feedback

        output = f"\nInternal feedback: {first_result}\n\nExternal feedback: {second_result}\n\nProcessed feedback: {end_result}"

        return output


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
