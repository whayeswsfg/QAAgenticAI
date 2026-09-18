from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langchain.messages import HumanMessage, AIMessage, SystemMessage
from langchain.agents.middleware import SummarizationMiddleware



# Load environment variables from .env
load_dotenv()

thread_config = {"configurable": {"thread_id": "stm"}}

llm_model = init_chat_model(
    model="gpt-5-nano",
    temperature=0.5,
    timeout=600,
    max_tokens=30000,
    max_retries=3,
)

summarize = SummarizationMiddleware(
    llm_model,
    trigger=("messages",8), #this is the trigger for the summarization middleware, it will summarize the conversation every 8 messages
    keep=("messages",2) #this is the number of messages to keep in the conversation history after summarization
)


#the purpose of the system prompt is to set the context for the agent's behavior and responses.
sys_msg = SystemMessage(
    content="You are helpful primary math teacher. "
)


my_agent = create_agent(
    model=llm_model,
    system_prompt=sys_msg,
    middleware=[summarize],
    checkpointer=InMemorySaver()
)

msg = [
    "What is 2+2?",
    "What is 8+14?",
    "What is 3+16?",
    "What is 4+4?"
    "What is 1+2?",
    "What is 100+14?",
    "What is 814+316?",
    "What is 8+14?"

]


for m in msg:
    response = my_agent.invoke({"messages": [HumanMessage(content=m)]}, thread_config)
    print(f"the response: {response}")
    print(f"the messages: {len(response['messages'])}")

