import streamlit as st
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun,DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper,ArxivAPIWrapper
#Custom tools
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain.prompts import ChatPromptTemplate
from langchain.agents import create_openai_tools_agent,initialize_agent,AgentType
from langchain.agents import AgentExecutor
from langchain.callbacks import StreamlitCallbackHandler
from langchain_openai import ChatOpenAI
import os

st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter API Key",type='password')

os.environ["OPENAI_API_KEY"] = api_key


#ArxivWrapper and wikipedia

api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=2,doc_content_chars_max=400)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)

arxiv_wrapper = ArxivAPIWrapper(top_k_results=2,doc_content_chars_max=400)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

search = DuckDuckGoSearchRun(name="RAMU")


st.title("Langchain - Chat with Search")



if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role":"assistant","content":"Hi, I'm chatbot who can search the web . How can I help you?"}
    ]

print(st.session_state.messages)
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
    





if prompt:=st.chat_input(placeholder="What is machine learning?"):
    st.session_state.messages.append({"role":"user","content":prompt})
    print("***",st.session_state.messages)

    st.chat_message("user").write(prompt)
    


    llm = ChatOpenAI(model='gpt-3.5-turbo',streaming=True)


    tools = [search,arxiv,wiki]

    search_agent = initialize_agent(tools,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,handling_parsing_error=True)

    with st.chat_message("assistant"):
        print("===>",st.session_state.messages)
        st_cb = StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
        print("%%%%%%%%",st.session_state.messages)

        response = search_agent.run(st.session_state.messages,callbacks=[st_cb])
        print(response)

        st.session_state.messages.append({'role':'assistant',"content":response})

        print("&&&&&",st.session_state.messages)


        st.write(response)


