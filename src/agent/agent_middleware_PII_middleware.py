from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langchain.messages import HumanMessage, AIMessage, SystemMessage
from langchain.agents.middleware import SummarizationMiddleware
from langchain.agents.middleware import HumanInTheLoopMiddleware, ModelCallLimitMiddleware, ToolCallLimitMiddleware,ModelFallbackMiddleware, PIIMiddleware
from langgraph.types import Command

#limit the number of calls to the model to save cost
#limit the number of calls to a tool to prevent infinite loops - email creation, api calls, etc.
# Load environment variables from .env
load_dotenv()

thread_config = {"configurable": {"thread_id": "stm"}}


def read_email_tool(email_id: str) -> str:
    """This tool reads email"""
    return f"Eail content for ID: {email_id}"

def send_email_tool(recipient: str, body: str="Email body") -> str:
    """This tool sends email"""
    return f"Email sent to: {recipient} with {body}"

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

hitl = HumanInTheLoopMiddleware(
    interrupt_on={"send_email_tool": {"allowed_decisions": ["approve", "reject"]}}
)

mclm = ModelCallLimitMiddleware(
    thread_limit=10,
    run_limit = 5,
    exit_behavior='end'
)

#Global tool call limit
gtclm = ToolCallLimitMiddleware(
    thread_limit=10,
    run_limit=5
)

#Tool Specific tool call limit
tstclm = ToolCallLimitMiddleware(
    tool_name=send_email_tool,
    thread_limit=10,
    run_limit=5
)

mfm = ModelFallbackMiddleware(
    "gpt-5.4-mini"
)

piimw = PIIMiddleware(
    pii_type="email",
    strategy="mask",
    apply_to_input=True
)



#the purpose of the system prompt is to set the context for the agent's behavior and responses.
sys_msg = SystemMessage(
    content="You are helpful assistant. "
)


my_agent = create_agent(
    model=llm_model,
    system_prompt=sys_msg,
    tools=[read_email_tool, send_email_tool],
    middleware=[hitl,mclm,gtclm,tstclm,mfm,piimw],
    checkpointer=InMemorySaver()
)

result = my_agent.invoke(
    {
        "messages": [
            (
                "user",
                "Read email 234 and send good morning Junie to junie@hotmail.com"
            )
        ]
    },
    config=thread_config,
)

print(result)

if "__interrupt__" in result:
    user_decision = input("Do you approve to send the email? (Approve/Reject):  ").lower().strip()

    result = my_agent.invoke(
        Command(resume={"decisions": [{"type": user_decision}]}),
        config = thread_config
    )
print(result["messages"][-1].content)

