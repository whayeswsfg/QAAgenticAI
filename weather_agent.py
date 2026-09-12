from langchain.agents import create_agent
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

agent = create_agent(
    model="openai:gpt-5.5",
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather in London, England?"}]}
)
print(result["messages"][-1].content_blocks)