from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.chat_models import init_chat_model

# Load environment variables from .env
load_dotenv()



llm_model = init_chat_model(
    model="gpt-5-nano",
    temperature=0.5,
    timeout=600,
    max_tokens=30000,
    max_retries=3,
)


my_agent = create_agent(
    model=llm_model,

    system_prompt="""You are a helpful assistant. """
)


user_msg1 = {
    "messages": [
        {
            "role": "user",
            "content": "My Name is Wanda Hayes."
        }
    ]
}

user_msg2 = {
    "messages": [
        {
            "role": "user",
            "content": "What is my name?"
        }
    ]
}

print('-' *80)
for chunk, metadata in my_agent.stream(user_msg1, stream_mode="messages"):
    print(chunk.content, end="", flush=True)
print('\n' + '-' *80)

print('*' *80)
for chunk, metadata in my_agent.stream(user_msg2, stream_mode="messages"):
    print(chunk.content, end="", flush=True)
print('\n' + '*' *80)