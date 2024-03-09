from dotenv import load_dotenv
load_dotenv()

from typing import Literal 

from langchain.agents import initialize_agent, load_tools, AgentType 
from langchain.chains.base import Chain
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain_community.chat_models import ChatLiteLLM
from langchain_experimental.plan_and_execute import (
    load_chat_planner, load_agent_executor, PlanAndExecute
)

ReasoningStrategies = Literal["zero-shot-react", "plan-and-solve"]

def load_agent(
    tool_names: list[str],
    strategy: ReasoningStrategies = "zero-shot-react",
) -> Chain:
    llm = ChatLiteLLM(model="gemini/gemini-pro")
    tools = load_tools(
        tool_names=tool_names,
        llm = llm
    )
    
    if strategy == "plan-and-solve":
        planner = load_chat_planner(llm)
        executor = load_agent_executor(llm, tools, verbose=True)
        return PlanAndExecute(planner=planner, executor=executor, verbose=True)
    
    return initialize_agent(
        tools=tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True,
    )