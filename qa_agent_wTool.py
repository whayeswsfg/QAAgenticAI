from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages.tool import tool_call
from langchain.tools import tool
from langchain.chat_models import init_chat_model

# Load environment variables from .env
load_dotenv()

@tool
def check_jira_for_login_defects():
    """This tool checks Jira for any open defects related to login functionality."""
    # This function would contain logic to check Jira for any existing defects related to login functionality.
    # For demonstration purposes, we'll return a placeholder string.
    return "Checked Jira for login defects. Found 1 high priority defect (nothing happens when the user clicks the login button)."

llm_model = init_chat_model(
    model="gpt-5-nano",
    temperature = 0.7, #this is to determine the randomness of the model's responses. A lower temperature (closer to 0) will make the model's responses more deterministic and focused, while a higher temperature (closer to 1) will make the responses more random and creative.
    timeout=600, #this is in seconds and it is to set the maximum time the model will take to respond. If the model takes longer than this time, it will raise a timeout error.
    max_tokens = 30000, #this is to limit the maximum number of tokens the model can generate in a single response. This helps to control the length of the output and manage costs associated with token usage.
    max_retries = 3, #this is to set the maximum number of times the model will retry generating a response if it encounters an error or fails to generate a response. This helps to improve reliability in case of transient issues.
  #  streaming=True, #this is to enable streaming of the model's responses. When set to True, the model will send partial responses as they are generated, allowing for a more interactive experience. If set to False, the model will wait until the entire response is generated before sending it back.
)


my_agent = create_agent(
    model=llm_model,
    #the purpose of a system prompt is to set the behavior and context for the agent. It defines the role, expertise, and tone of the agent, guiding how it should respond to user queries. In this case, it establishes the agent as an expert in QA (Quality Assurance) AI engineering, ensuring that its responses are relevant and informed by that expertise.
    system_prompt="You are an expert QA AI Engineer.  You are a Lead Test Automation Engineer.  You use Katalon with groovy as your scripting language. You have Insurance and Annuity Business Domain Knowlede",
    tools=[check_jira_for_login_defects]
)

result = my_agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Are there any open defect in jira? if yes, write a test case for it."
            }
        ]
    }
)

#print(result)
#print(result["messages"][-1].content_blocks)
print(result["messages"][-1].content)