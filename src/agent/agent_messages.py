from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langchain.messages import HumanMessage, AIMessage, SystemMessage


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

#the purpose of the system prompt is to set the context for the agent's behavior and responses.
sys_msg = SystemMessage(
    content="You are a helpful QA SDET assistant that specializes in API and UI Automation.  "
)


my_agent = create_agent(
    model=llm_model,
    system_prompt=sys_msg,
    checkpointer=InMemorySaver()
)


human_msg1 = HumanMessage(
    content="My Name is Junie."
)
ai_msg1 = AIMessage(
    content="Hello Junie! How can I assist you today?"
)

human_msg2 = HumanMessage(
    content="What is my name?"
)
ai_msg2 = AIMessage(
    content="Your name is Junie."
)

human_msg3 = HumanMessage(
    content="Can you write API Automation Test Scenarios?"
)
ai_msg3 = AIMessage(
    content="Yes"
)

human_msg4 = HumanMessage(
    content="Can you write UI Automation Test Scenarios?"
)
ai_msg4 = AIMessage(
    content="Yes"
)

human_message = HumanMessage(content="What is my name? What is a QA SDET that specializes in Automation?")


print('-' *80)
for chunk, metadata in my_agent.stream({"messages": [human_msg1,
                                                     ai_msg1,
                                                     human_msg2,
                                                     ai_msg2,
                                                     human_msg3,
                                                     ai_msg3,
                                                     human_msg4,
                                                     ai_msg4,
                                                     human_message]},thread_config,stream_mode="messages"):
    print(chunk.content, end="", flush=True)
print('\n' + '-' *80)

