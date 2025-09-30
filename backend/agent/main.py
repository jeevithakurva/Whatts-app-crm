import json
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.llms import Bedrock
from langchain.agents import initialize_agent, AgentType
from langchain_community.agent_toolkits.openapi.toolkit import RequestsToolkit
from langchain_community.utilities.openapi import OpenAPISpec

# Load OpenAPI spec from file as raw text
with open("wotnot_openapi.json", "r") as f:
    openapi_raw = f.read()

spec = OpenAPISpec.from_text(openapi_raw)
spec.base_url = "https://api.wotnot.io"

# Get tools from the OpenAPI spec
toolkit = RequestsToolkit(spec=spec)
tools = toolkit.get_tools()

# Initialize LLM
llm = Bedrock(
    model_id="anthropic.claude-v2",
    region_name="us-east-1",
    model_kwargs={"temperature": 0.2}
)

# Initialize LangChain agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True
)

# FastAPI app initialization with CORS (allow all for development)
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model for the prompt input
class PromptRequest(BaseModel):
    prompt: str

@app.post("/run-agent/")
async def run_agent(request: Request):
    body = await request.json()
    prompt = body.get("prompt", "Find content for template, create it, and send to today's contacts.")
    try:
        # Run the LangChain agent with the prompt
        result = agent.run(prompt)
        return {"response": result}
    except Exception as e:
        return {"error": str(e)}

# New endpoint to generate predefined message based on user prompt
@app.post("/generate-message/")
async def generate_message(data: PromptRequest):
    prompt = data.prompt
    try:
        # You can customize how to call the agent with the prompt for generation
        generated_message = agent.run(prompt)
        return {"message": generated_message}
    except Exception as e:
        return {"error": str(e)}
