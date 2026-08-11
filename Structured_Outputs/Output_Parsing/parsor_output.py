from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

# StructuredOutputParser -> forces output into SPECIFIC named fields
# ResponseSchema -> defines each field's name + description (like a mini schema)
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

load_dotenv()

# Connect to a Hugging Face model (Gemma)
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

# Wrap it so it behaves like a chat model
model = ChatHuggingFace(llm=llm)

# Define the exact fields we want in the output
# Each ResponseSchema = one field name + a description telling the AI what to put there
schema = [
    ResponseSchema(name='fact_1', description='Fact 1 about the topic'),
    ResponseSchema(name='fact_2', description='Fact 2 about the topic'),
    ResponseSchema(name='fact_3', description='Fact 3 about the topic'),
]

# Build the parser from our field definitions
parser = StructuredOutputParser.from_response_schemas(schema)

# Build the prompt template
# format_instruction auto-fills with instructions telling the AI exactly
# what fields to return and in what format (based on our schema above)
template = PromptTemplate(
    template='Give 3 fact about {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

# Chain: template -> model -> parser
# 1. template fills {topic}, builds the prompt
# 2. model sends prompt to LLM, gets back a formatted text reply
# 3. parser reads that reply and converts it into a dict with
#    exactly the keys: fact_1, fact_2, fact_3
chain = template | model | parser

# Run the pipeline
result = chain.invoke({'topic': 'black hole'})

# result is a dict like: {'fact_1': '...', 'fact_2': '...', 'fact_3': '...'}
print(result)
