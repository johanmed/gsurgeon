"""
Multi-agent system to dissect genomic information
Main module of the package
Author: Johannes Medagbe
Copyright (c) 2026
"""

import json
import logging
import os
import time
import warnings
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from prompts import (
    expert_prompt,
    planner_prompt,
    reflector_prompt,
    researcher_prompt,
    supervisor_prompt1,
    supervisor_prompt2,
)
from tools import AgentState, Consult, Finalize, Plan, Research, Supervise, Tune

warnings.filterwarnings("ignore")


@dataclass
class GSurgeon:
    """
    Represent Search Agent
    Input:
        max_iterations: maximum number of iterations allowed
    Operations:
        Initialization of multi-agent graph
        Run of query through system
    """

    max_iterations: int = 10

    def _researcher(self, state: AgentState) -> dict:
        if len(state.messages) < 3:  # handle first call to researcher
            input_text = state.messages[0]  # use original query
        else:
            input_text = state.messages[-1]  # use reflection insights
        input_text = [researcher_prompt, input_text.content]
        research = Research()
        result = research(query=input_text)
        return {
            "messages": [result.get("solution")],
        }

    def _expert(self, state: AgentState) -> dict:
        if len(state.messages) < 4:  # handle first call to expert
            input_text = state.messages[1] + state.messages[0]  # use plan and query
        else:
            input_text = state.messages[-2]  # use reflection insights
        input_text = [expert_prompt, input_text]
        consult = Consult()
        result = consult(query=input_text)
        return {
            "messages": [result.get("solution")],
        }

    def _planner(self, state: AgentState) -> dict:
        plan = dspy.Predict(Plan)
        input_text = [planner_prompt] + state.messages
        result = plan(background=input_text)
        return {
            "messages": [result.get("answer")],
        }

    def reflector(self, state: AgentState) -> dict:
        tune = dspy.Predict(Tune)
        trans_map = {AIMessage: HumanMessage, HumanMessage: AIMessage}
        translated_messages = [refl_prompt, state.messages[0]] + [
            trans_map[msg.__class__](content=msg.content) for msg in state.messages[1:]
        ]
        result = tune(background=translated_messages)
        return {
            "messages": [
                HumanMessage(
                    f"Progress has been made. Use now all the resources to addess this new suggestion: {result.get('answer')}"
                )
            ],
        }

    def _supervisor(self, state: AgentState) -> dict:
        supervise = dspy.Predict(Supervise)
        messages = [
            ("system", self.sup_prompt1),
            *state.messages,
            ("system", self.sup_prompt2),
        ]
        if len(messages) > self.max_iterations:
            return {"next_decision": "end"}
        result = supervise(background=messages)
        return {
            "next_decision": result.get("next_decision"),
        }

    async def _run_graph(self, query: str) -> Any:
        graph_builder = StateGraph(AgentState)
        graph_builder.add_node("researcher", self._researcher)
        graph_builder.add_node("planner", self._planner)
        graph_builder.add_node("reflector", self._reflector)
        graph_builder.add_node("supervisor", self._supervisor)
        graph_builder.add_node("expert", self._expert)
        graph_builder.add_edge(START, "planner")
        graph_builder.add_edge("researcher", "supervisor")
        graph_builder.add_edge("expert", "supervisor")
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
        initial_state = {
            "messages": [("human", query)],
            "next_decision": "planner",  # always plan first
        }
        result = await graph.ainvoke(initial_state)
        return result

    async def handle(self, query: str) -> str:
        result = await self._run_graph(query)
        unprocessed_result = result.get("messages")[
            2
        ].content
        finalize = dspy.Predict(Finalize)
        processed_result = finalize(messages=result.get("messages")).get("feedback")
        output = f"Raw feedback: {unprocessed_result}\nProcessed feedback: {processed_result}"
        return output
