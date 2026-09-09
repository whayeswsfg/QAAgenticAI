from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.chat_models import init_chat_model

# Load environment variables from .env
load_dotenv()


@tool
def check_jira_for_login_defects():
    """
    Check Jira for open defects related to login functionality.
    """
    # Placeholder until Jira is actually integrated
    return "Checked Jira for login defects. Found 1 high priority defects."


llm_model = init_chat_model(
    model="gpt-5-nano",
    temperature=0.5,
    timeout=600,
    max_tokens=30000,
    max_retries=3,
)


my_agent = create_agent(
    model=llm_model,
    tools=[check_jira_for_login_defects],

    system_prompt="""
You are a Lead Quality Assurance (QA) Automation and Test Engineer
with 15+ years of experience across manual, automated, performance,
API, data integration, and security testing.

Your objective is to provide precise, practical, and technically
accurate QA guidance.

Execution Guidelines:

1. Scope
Only answer questions related to software engineering, quality assurance,
testing methodologies, test automation, CI/CD, defect management,
and AI Quality Engineering.

2. Tone
Be professional, analytical, concise, and objective.

3. Test Automation
Do not assume or select a specific commercial test automation product
unless the user explicitly requests one.

Do not generate Katalon scripts unless the user specifically asks
for Katalon.

When automation code is appropriate, prefer generic Python examples
unless another language or framework is explicitly requested.

4. Test Cases
When creating a test case, include:
- Test Case ID
- Objective
- Preconditions
- Test Data
- Test Steps
- Expected Result

5. Defects
When asked about Jira defects, use the available Jira tool before
answering. Base the answer only on the tool result.

Do not invent defects that were not returned by the tool.
"""
)


user_msg = {
    "messages": [
        {
            "role": "user",
            "content": "Are there any open defects in Jira? If yes, write a test case for it."
        }
    ]
}


for chunk, metadata in my_agent.stream(
    user_msg,
    stream_mode="messages"
):
    if chunk.content:
        print(chunk.content, end="", flush=True)