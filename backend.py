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
    llm_call: str
    itinerary: atr
