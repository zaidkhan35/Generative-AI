from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

# JsonOutputParser -> tells the model to return JSON, and parses that JSON
# into a real Python dict automatically
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

# Connect to a Hugging Face model (Gemma)
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

# Wrap it so it behaves like a chat model
model = ChatHuggingFace(llm=llm)

# Create the JSON parser
# This parser does 2 jobs:
# 1. get_format_instructions() -> gives text instructions telling the AI
#    exactly HOW to format its JSON output
# 2. Later (after model replies) -> reads that JSON text and converts it
#    into an actual Python dict
parser = JsonOutputParser()

# Build the prompt template
# {format_instruction} is auto-filled using partial_variables (fixed value,
# doesn't need to be passed in every time we invoke)
template = PromptTemplate(
    template='Give me 5 facts about {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

# ---------------------------------------------------------
# THE CHAIN: template | model | parser
# ---------------------------------------------------------
# This is a PIPELINE built using the | (pipe) operator.
# Data flows left to right, automatically, through 3 stages:
#
# Stage 1 (template): takes {'topic': 'black hole'} as input,
#                      fills the placeholder, produces a final PROMPT (text)
#
# Stage 2 (model):     takes that prompt as input, sends it to the LLM,
#                      LLM replies with a JSON-formatted STRING (raw text)
#
# Stage 3 (parser):    takes that JSON string as input, parses/converts it
#                      into an actual Python DICT you can use in code
#
# So instead of manually doing: fill template -> call model -> parse JSON
# yourself in 3 separate lines, the "|" does it all in ONE flowing pipeline
chain = template | model | parser

# Run the whole pipeline with just the topic
# (format_instruction is already baked in via partial_variables)
result = chain.invoke({'topic': 'black hole'})

# result is now a Python dict/list (NOT a string) -> ready to use directly
print(result)
