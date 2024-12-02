import os
import itertools

from langchain import hub  
from langchain_core.tools import Tool
from langchain.agents import AgentExecutor
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent
from langchain_core.runnables.base import RunnableEach
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

search = GoogleSearchAPIWrapper()

def top5Results(query):
    return search.results(query, 3)


tool = Tool(
    name="google_search",
    description="Search Google for recent results.",
    func=top5Results,
)

memory = MemorySaver()
search = TavilySearchResults(max_results=5)
tools = [search]

model = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro", google_api_key=os.environ.get("GEMINI_API_KEY")
)

prompt = hub.pull("hwchase17/openai-functions-agent")
agent = create_tool_calling_agent(model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)
agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

config = {"configurable": {"session_id": "abc123"}}


def append_query(query):
    return f"Execute the following query on the search tool and summarize its results: {query}"


retrieve_and_summarize = {"input": RunnablePassthrough()} | agent_with_chat_history

create_quiz_chain = (
    ChatPromptTemplate(
        [
            (
                "human",
                (
                    "Create seven multiple choice quiz based on the following text.\n"
                    "Please make the questions be as broad as possible\n"
                    "Output in JSON with following keys:\n"
                    "question: The question\n"
                    "options: four possible options\n"
                    "correct_answer_index: Index of the correct option\n"
                    "{output}"
                ),
            ),
        ]
    )
    | model
    | JsonOutputParser()
)


def clean_result(result):
    return result["output"]


chain = (
    ChatPromptTemplate(
        [
            (
                "human",
                (
                    "I want to create a Quiz on {query},"
                    "Give me five search queries to find suprising things about it. Output as a JSON List"
                ),
            ),
        ]
    )
    | model
    | JsonOutputParser()
    | RunnableEach(bound=RunnableLambda(append_query))
    | RunnableEach(bound=retrieve_and_summarize)
    | RunnableEach(bound=RunnableLambda(clean_result))
    | RunnableEach(bound=create_quiz_chain)
    | RunnableLambda(lambda x: list(itertools.chain.from_iterable(x)))
)
