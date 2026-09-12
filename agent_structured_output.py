from dotenv import load_dotenv
from langchain.agents import create_agent

# Load environment variables from .env
load_dotenv()

my_agent = create_agent(
    model="gpt-5-nano",
    #the purpose of a system prompt is to set the behavior and context for the agent. It defines the role, expertise, and tone of the agent, guiding how it should respond to user queries. In this case, it establishes the agent as an expert in QA (Quality Assurance) AI engineering, ensuring that its responses are relevant and informed by that expertise.
    system_prompt="You are an expert QA AI Engineer.  You are a Lead Test Automation Engineer.  You use Katalon with groovy as your scripting language. You have Insurance and Annuity Business Domain Knowlede"
)

result = my_agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Write a test scenaarios for login functionality of a web application.  Write test Katalon automted test scripts in groovy for the test scenarios.  Write the test scripts in a way that they can be run in Katalon Studio. Create the script output in .csv file format ready for download"

            }
        ]
    }
)

#print(result)
#print(result["messages"][-1].content_blocks)
print(result["messages"][-1].content)