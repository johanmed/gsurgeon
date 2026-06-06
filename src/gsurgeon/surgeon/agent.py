"""GSurgeon: Multi-agent system to dissect genomic information"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

import dspy
from gsurgeon.procedures.react import Consult, Research
from gsurgeon.procedures.standard import Finalize, Plan, Supervise, Tune
from gsurgeon.surgeon.prompts import (
    expert_prompt,
    planner_prompt,
    reflector_prompt,
    researcher_prompt,
    supervisor_prompt1,
    supervisor_prompt2,
)
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel
from typing_extensions import Annotated, Literal


class AgentState(BaseModel):
    """
    Represent agent state
    Avail 02 attributes to allow communication between agents
    """

    messages: Annotated[list[BaseMessage], add_messages]
    next_decision: Literal[
        "gn_researcher", "planner", "reflector", "ncbi_expert", "end"
    ]


@dataclass
class GSurgeon:
    """
    Represent Search Agent
    Input:
        max_steps: maximum number of steps allowed
    Operations:
        Initialization of multi-agent graph
        Run of query through system
    """

    max_steps: int = 5
    _graph: Any = field(init=False)

    def __post_init__(self):
        self._graph = self._build_graph()

    async def _researcher(self, state: AgentState) -> dict:
        """Answer query with GeneNetwork information"""
        print("Calling the researcher...")
        if len(state.messages) < 3:  # handle first call to researcher
            input_text = state.messages[0]  # use original query
        else:
            input_text = state.messages[-1]  # use reflection insights
        input_text = [researcher_prompt, input_text.content]
        research = Research()
        result = await asyncio.to_thread(research, query=input_text)
        print("Researcher performed analysis")
        return {
            "messages": [AIMessage(result.get("solution"))],
        }

    async def _expert(self, state: AgentState) -> dict:
        """Answer query with NCBI information"""
        print("Calling the expert...")
        if len(state.messages) < 4:  # handle first call to expert
            input_text = state.messages[1] + state.messages[0]  # use plan and query
        else:
            input_text = state.messages[-2]  # use reflection insights
        input_text = [expert_prompt, input_text]
        consult = Consult()
        result = await asyncio.to_thread(consult, query=input_text)
        print("Expert produced answers")
        return {
            "messages": [AIMessage(result.get("solution"))],
        }

    async def _planner(self, state: AgentState) -> dict:
        """Generate a plan to solve query"""
        print("Generating a plan to solve the problem...")
        plan = dspy.Predict(Plan)
        input_text = [planner_prompt] + state.messages
        result = await asyncio.to_thread(plan, background=input_text)
        print("Plan acquired")
        return {
            "messages": [AIMessage(result.get("answer"))],
        }

    async def _reflector(self, state: AgentState) -> dict:
        """Propose improvements to answer"""
        print("Calling the reflector...")
        tune = dspy.Predict(Tune)
        trans_map = {AIMessage: HumanMessage, HumanMessage: AIMessage}
        translated_messages = [reflector_prompt, state.messages[0]] + [
            trans_map[msg.__class__](content=msg.content) for msg in state.messages[1:]
        ]
        result = await asyncio.to_thread(tune, background=translated_messages)
        print("Reflector made suggestions")
        return {
            "messages": [
                HumanMessage(
                    f"Progress has been made. Use now all the resources to addess this new suggestion: {result.get('answer')}"
                )
            ],
        }

    async def _supervisor(self, state: AgentState) -> dict:
        """Orchestrate agentic system"""
        print("Getting guidance from the supervisor...")
        supervise = dspy.Predict(Supervise)
        messages = [
            supervisor_prompt1,
            *state.messages,
            supervisor_prompt2,
        ]
        if len(messages) > self.max_steps:
            return {"next_decision": "end"}
        result = await asyncio.to_thread(supervise, background=messages)
        print("Supervisor selected the next worker")
        return {
            "next_decision": result.get("next_decision"),
        }

    def _build_graph(self) -> Any:
        graph_builder = StateGraph(AgentState)
        graph_builder.add_node("gn_researcher", self._researcher)
        graph_builder.add_node("planner", self._planner)
        graph_builder.add_node("reflector", self._reflector)
        graph_builder.add_node("supervisor", self._supervisor)
        graph_builder.add_node("ncbi_expert", self._expert)
        graph_builder.add_edge(START, "planner")
        graph_builder.add_edge("planner", "supervisor")
        graph_builder.add_edge("gn_researcher", "supervisor")
        graph_builder.add_edge("ncbi_expert", "supervisor")
        graph_builder.add_edge("reflector", "supervisor")
        graph_builder.add_conditional_edges(
            "supervisor",
            lambda state: state.next_decision,
            {
                "reflector": "reflector",
                "gn_researcher": "gn_researcher",
                "ncbi_expert": "ncbi_expert",
                "end": END,
            },
        )
        return graph_builder.compile()

    async def _run_graph(self, query: str) -> Any:
        initial_state = {
            "messages": [HumanMessage(query)],
            "next_decision": "planner",  # always plan first
        }
        return await self._graph.ainvoke(initial_state)

    async def handle(self, query: str) -> str:
        """Run query through the system"""
        print("Starting operation...")
        result = await self._run_graph(query)
        unprocessed_result = result.get("messages")[2].content
        finalize = dspy.Predict(Finalize)
        processed_result = await asyncio.to_thread(
            lambda: finalize(messages=result.get("messages")).get("feedback")
        )
        print("Operation complete")
        return f"Raw feedback: {unprocessed_result}\nProcessed feedback: {processed_result}"
