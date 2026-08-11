from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# RunnableParallel -> runs multiple chains at the same time
# RunnableBranch -> picks ONE path to run based on a condition (like if/elif/else)
# RunnableLambda -> wraps a plain Python function so it can be used inside a chain
from langchain.schema.runnable import RunnableParallel, RunnableBranch, RunnableLambda

from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

# Set up the model
model = ChatOpenAI()

# Plain text parser (used later for the actual response)
parser = StrOutputParser()

# Schema to classify feedback sentiment - only allows 'positive' or 'negative'
class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description='Give the sentiment of the feedback')

# Parser that validates and converts model output into a Feedback object
parser2 = PydanticOutputParser(pydantic_object=Feedback)

# Prompt to classify feedback as positive/negative
prompt1 = PromptTemplate(
    template='Classify the sentiment of the following feedback text into postive or negative \n {feedback} \n {format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction': parser2.get_format_instructions()}
)

# Chain to classify sentiment: feedback text -> Feedback object (sentiment field)
classifier_chain = prompt1 | model | parser2

# Prompt to respond to POSITIVE feedback
prompt2 = PromptTemplate(
    template='Write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)

# Prompt to respond to NEGATIVE feedback
prompt3 = PromptTemplate(
    template='Write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)

# ---------------------------------------------------------
# RunnableBranch: like an if/elif/else for chains
# ---------------------------------------------------------
# Takes the OUTPUT of classifier_chain (a Feedback object) as input (x)
# Checks conditions one by one, top to bottom:
# - if x.sentiment == 'positive' -> run prompt2 | model | parser (positive response)
# - elif x.sentiment == 'negative' -> run prompt3 | model | parser (negative response)
# - else (no match) -> RunnableLambda runs a plain Python function as fallback
branch_chain = RunnableBranch(
    (lambda x: x.sentiment == 'positive', prompt2 | model | parser),
    (lambda x: x.sentiment == 'negative', prompt3 | model | parser),
    RunnableLambda(lambda x: "could not find sentiment")  # fallback if nothing matches
)

# Full chain: classify sentiment first, THEN branch to the correct response chain
# based on what sentiment was detected
chain = classifier_chain | branch_chain

# Run the pipeline with sample feedback
print(chain.invoke({'feedback': 'This is a beautiful phone'}))

# Print ASCII diagram of the chain
# will show a branching structure: classify -> then splits into
# positive-path / negative-path / fallback-path
chain.get_graph().print_ascii()
