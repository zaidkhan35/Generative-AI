from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

# PydanticOutputParser -> parses AI's text output into a validated Pydantic object
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

# Connect to a Hugging Face model (Gemma)
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

# Wrap it so it behaves like a chat model
model = ChatHuggingFace(llm=llm)

# Define the schema using Pydantic (with real validation rules)
class Person(BaseModel):
    name: str = Field(description='Name of the person')

    # gt=18 -> age MUST be greater than 18, Pydantic will enforce this
    age: int = Field(gt=18, description='Age of the person')

    city: str = Field(description='Name of the city the person belongs to')

# Create the parser using our Person schema
parser = PydanticOutputParser(pydantic_object=Person)

# Build the prompt template
# format_instruction auto-fills with instructions telling the AI exactly
# what fields and format to return, based on the Person schema
template = PromptTemplate(
    template='Generate the name, age and city of a fictional {place} person \n {format_instruction}',
    input_variables=['place'],

    #The output should be formatted as a JSON instance that conforms to the JSON schema below
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

# Chain: template -> model -> parser
# 1. template fills {place}, builds the prompt
# 2. model sends prompt to LLM, gets back text (formatted like the schema)
# 3. parser reads that text, converts it into a validated Person object
#    (throws an error if data doesn't match the schema, e.g. age <= 18)
chain = template | model | parser

# Run the pipeline
final_result = chain.invoke({'place': 'sri lankan'})

# final_result is a Person object (not a dict)
# access fields with dot notation: final_result.name, final_result.age, etc.
print(final_result)
