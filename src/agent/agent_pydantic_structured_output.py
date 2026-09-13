from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages.tool import tool_call
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

# Load environment variables from .env
load_dotenv()

@tool
def check_jira_for_login_defects():
    """This tool checks Jira for any open defects related to login functionality."""
    # This function would contain logic to check Jira for any existing defects related to login functionality.
    # For demonstration purposes, we'll return a placeholder string.
    return "Found 1 high priority defect in Jira related to login functionality: (Nothing happens when the user clicks the login button)."

#Define schema for test case output
class TestCaseOutput(BaseModel):
    """Structured test case output for the defect found in Jira."""
    test_case_id: int = Field(description="Unique test case ID")
    test_case_title: str = Field(description="Short name of the test case.  It is descriptive of the defect found in Jira.")
    test_case_description: str = Field(description="Short summary of the test case that describes the defect found in Jira.")
    precondition: list[str] = Field(description="Test case detailed pre-requisites before executing the test case.  It is detailed list of preconditions that need to be met before executing the test case.")
    steps_to_reproduce: str = Field(description="Step by step detailed description to execute the test case.  \
                                                The steps are strictly formatted as a numbered list.  Each step is a single action that the end user will take to execute the test case.  \
                                                The steps are written in a way a developer or another tester can follow.")
    expected_result: str = Field(description="What is expected behavior by the end user was using the application.")
    actual_result: str = Field(description="What is the current behavior due to the defect.")




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
    system_prompt="You are an Senior Software Development Engineer in Test that specializes in Automation Engineer.  When there is a defect found in Jira, you will write a test case for it.  You will write the test case in a structured format using pydantic schema.  You will use the TestCaseOutput schema to write the test case. ",
    response_format=TestCaseOutput,
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

structured_resp: TestCaseOutput = result["structured_response"]

print(f"Test Case ID: {structured_resp.test_case_id}\n")
print(f"Test Case Title: {structured_resp.test_case_title}\n")
print(f"Test Case Description: {structured_resp.test_case_description}\n")
print(f"Precondition: {structured_resp.precondition}\n")
print(f"Steps to Reproduce:\n{structured_resp.steps_to_reproduce}\n")
print(f"Expected Result: {structured_resp.expected_result}\n")
print(f"Actual Result: {structured_resp.actual_result}\n")



