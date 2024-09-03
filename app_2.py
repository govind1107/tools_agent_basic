import streamlit as st
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun,DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper,ArxivAPIWrapper
#Custom tools
from langchain_community.document_loaders import WebBaseLoader,PyPDFLoader
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
from langchain.agents import AgentExecutor
from langchain.agents import create_openai_tools_agent





os.environ["OPENAI_API_KEY"] = "Your API Key"

def agent_m():

    print("Inside function")
    loader_web = WebBaseLoader("https://naruto.fandom.com/wiki/Naruto_(series)")
    docs_web = loader_web.load()
    docs_book = PyPDFLoader('attention.pdf').load()
    print('Document loaded')
    text_splitter_web = RecursiveCharacterTextSplitter(chunk_size=2000,chunk_overlap=100)
    sp_web = text_splitter_web.split_documents(docs_web)
    text_splitter_book = RecursiveCharacterTextSplitter(chunk_size=2000,chunk_overlap=100)
    sp_book = text_splitter_book.split_documents(docs_book)
    print("Document chunked")

    retriever_web = FAISS.from_documents(sp_web,OpenAIEmbeddings()).as_retriever()
    retriever_book = FAISS.from_documents(sp_book,OpenAIEmbeddings()).as_retriever()


    print("Retriver created")
    retriever_tool_web = create_retriever_tool(retriever_web,"naruto_search","Search any info about naruto")
    retriever_tool_book = create_retriever_tool(retriever_book,"harrypotter_search","Search any info about harrypotter")

    return retriever_tool_web,retriever_tool_book


def web_arxiv():
    api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=2,doc_content_chars_max=400)
    wiki = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)

    arxiv_wrapper = ArxivAPIWrapper(top_k_results=2,doc_content_chars_max=400)
    arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

    return wiki,arxiv


def main_wrapper(query,tools):

    print("query")


    llm = ChatOpenAI(model='gpt-3.5-turbo',streaming=True)

    prompt = ChatPromptTemplate.from_messages(

        [
            ("system","You are an helpful assistant ,please response to below query"),
            ("user","Question : {question}"),
            ("placeholder", "{agent_scratchpad}")
        ]
    )

    agent = create_openai_tools_agent(llm,tools,prompt)


    agent_exec = AgentExecutor(agent=agent,tools=tools,verbose=True)

    return agent_exec.invoke({"question":f"{query}"})

    


st.title("Welcome to agentic rag")

user_input = st.text_input("Enter your query")

if st.button("submit"):
   retriever_tool_web,retriever_tool_book = agent_m()
   wiki,arxiv = web_arxiv()
   tools = [arxiv,wiki,retriever_tool_web,retriever_tool_book]

   response = main_wrapper(user_input,tools)

   st.write(response['output'])


   

        

















