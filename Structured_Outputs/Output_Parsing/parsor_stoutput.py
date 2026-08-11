# =====================================================
# VERSION 1: WITHOUT chaining/parser (manual step-by-step)
# =====================================================
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

# Connect to a Hugging Face model (Gemma)
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

# Wrap it so it behaves like a chat model
model = ChatHuggingFace(llm=llm)

# 1st prompt template -> ask for a detailed report
template1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

# 2nd prompt template -> ask for a 5-line summary of some text
template2 = PromptTemplate(
    template='Write a 5 line summary on the following text. /n {text}',
    input_variables=['text']
)

# Step 1: fill template1 with the topic
prompt1 = template1.invoke({'topic': 'black hole'})

# Step 2: manually send prompt1 to the model, get the detailed report back
result = model.invoke(prompt1)

# Step 3: manually take that report's text and fill it into template2
prompt2 = template2.invoke({'text': result.content})

# Step 4: manually send prompt2 to the model, get the summary back
result1 = model.invoke(prompt2)

# Print the final summary
print(result1.content)
# NOTE: every step here is done BY HAND -> you write .invoke() again and again
# and manually pass .content from one step into the next template


# =====================================================
# VERSION 2: WITH chaining + parser (automatic pipeline)
# =====================================================
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Set up OpenAI chat model
model = ChatOpenAI()

# Same two templates as before
template1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

template2 = PromptTemplate(
    template='Write a 5 line summary on the following text. /n {text}',
    input_variables=['text']
)

# StrOutputParser -> automatically extracts just the plain text (.content)
# from the model's reply, instead of you writing result.content manually
parser = StrOutputParser()

# Build a CHAIN using the | (pipe) operator
# This links steps together: output of one step automatically becomes
# input of the next step -> no manual .invoke() or .content needed in between
chain = template1 | model | parser | template2 | model | parser

# Run the ENTIRE pipeline in one call
# topic -> report generated -> parsed to text -> summarized -> parsed to text
result = chain.invoke({'topic': 'black hole'})

# Print the final summary
print(result)
