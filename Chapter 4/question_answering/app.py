import streamlit as st 
from langchain_community.callbacks import StreamlitCallbackHandler

from agent import load_agent
from utils import MEMORY 

st.set_page_config(page_title="LangChain Question Answering.", page_icon=":robot:")
st.header("Ask a research question!")

strategy = st.radio(
    "Reasoning strategy",
    ("plan-and-solve", "zero-shot-react",)
)

tool_names = st.multiselect(
    "Which tools do you want to use?",
    [
        "ddg-search", "arxiv", "wikipedia", "python_repl", "pal-math", "llm-math"
    ],
    ["ddg-search", "wikipedia"]
)

if st.sidebar.button("Clear message hitory"):
    MEMORY.chat_memory.clear()
    
avatars = {"human": "user", "ai": "assistant"}
for msg in MEMORY.chat_memory.messages:
    st.chat_message(avatars[msg.type]).write(msg.content)
    
assert strategy is not None
agent_chain = load_agent(tool_names=tool_names, strategy=strategy)

assistant = st.chat_message("assistant")
if prompt := st.chat_input(placeholder="Ask me anything!"):
    st.chat_message("user").write(prompt)
    stream_handler = StreamlitCallbackHandler(assistant)
    with st.chat_message("assistant"):
        response = agent_chain.run({
            "input":prompt,
            "chat_history": MEMORY.chat_memory.messages
        }, callbacks=[stream_handler]
                                   )