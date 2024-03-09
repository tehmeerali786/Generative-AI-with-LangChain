from dotenv import load_dotenv 
load_dotenv()

from langchain.agents import (
    AgentExecutor, AgentType, initialize_agent, load_tools
)

from langchain_google_genai import ChatGoogleGenerativeAI
def load_agent() -> AgentExecutor:
    llm = ChatGoogleGenerativeAI(model="gemini-pro")
    tools = load_tools(
        tool_names = ["ddg-search", "arxiv", "wikipedia"],
        llm = llm
    )
    return initialize_agent(
        tools=tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )
    
import streamlit as st
from langchain.callbacks import StreamlitCallbackHandler

chain = load_agent()
st_callback = StreamlitCallbackHandler(st.container())

if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        response = chain.run(prompt, callbacks=[st_callback])
        st.write(response)

 