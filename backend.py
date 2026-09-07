import os
import certifi
from dotenv import load_dotenv
from tools.flight_tool import search_flights
from tools.tavily_tool import tavily_search

load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where

import operator
import asyncio

from typing import Annotated, TypedDict
from langgraph.graph import START,END, StateGraph

from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)

from langchain_openai import ChatOpenAI

api_key=os.getenv("OPENAI_API_KEY")


# ============LLM===============

llm=ChatOpenAI(
    model="gpt-40-mini",
    api_key= api_key
)

#==================STATE================

class TravelState(TypedDict):
    user_query:str
    messages:Annotated[list[AnyMessage],operator.add]
    flight_results: str
    hotel_results : str
    llm_calls: str
    itinerary: str

#===========Flight agent =================

def flight_agent(state: TravelState):
    query=state["user_query"]

    flight_data=search_flights(query)

    return {
        "flight_results": flight_data,
        "messages": [
            AIMessage(content="Flight results fetched")
        ],

        "llm_calls": state.get("llm_call", 0) + 1


    }

def hotel_agent(state:TravelState):
    query=f"Best Hotels for {state['user_query']}"

    hotel_results= tavily_search(query)

    return {
        "hotel_results": hotel_results,
        "messages": [
            AIMessage(content=" Hotels detailed fetched")
        ],
        "llm_calls": state.get('llm_calls',0) + 1
    }

