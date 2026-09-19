from dotenv import load_dotenv
from langchain.agents import create_agent

# Load environment variables from .env
load_dotenv()

my_agent = create_agent(
    model="gpt-5-nano",
    #the purpose of a system prompt is to set the behavior and context for the agent. It defines the role, expertise, and tone of the agent, guiding how it should respond to user queries. In this case, it establishes the agent as an expert in QA (Quality Assurance) AI engineering, ensuring that its responses are relevant and informed by that expertise.
    system_prompt="you are a political analyst"
)

result = my_agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "fetch the latest from reliable sources and compile an up-to-date list of who is running for governor of ohio for 2026"

            }
        ]
    }
)

#print(result)
#print(result["messages"][-1].content_blocks)
print(result["messages"][-1].content)