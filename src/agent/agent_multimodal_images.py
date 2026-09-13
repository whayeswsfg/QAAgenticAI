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
    content="You are a agent that handles multiple modalities of input and output, including text, images, and other media. \
    You are a helpful QA SDET assistant that specializes in API and UI Automation.  "
)


my_agent = create_agent(
    model=llm_model,
    system_prompt=sys_msg,
    checkpointer=InMemorySaver()
)



from langchain_core.messages import HumanMessage

human_message = HumanMessage(
    content=[
        {
            "type": "text",
            "text": "Describe the image on the page."
        },
        {
            "type": "image_url",
            "image_url":
            {
                "url": "https://assets.zyrosite.com/cdn-cgi/image/format=auto,w=768,fit=crop,q=95/A1aP9M3K3ph4ZwZe/sqamedianew-3-PuLjr0Y473949E7z.png"
            }
        }
    ]
)


print('-' *80)
for chunk, metadata in my_agent.stream({"messages": [ human_message]},thread_config,stream_mode="messages"):
    print(chunk.content, end="", flush=True)
print('\n' + '-' *80)

