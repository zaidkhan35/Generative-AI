# Import tool to chat with OpenAI models using LangChain
from langchain_openai import ChatOpenAI

# Import tool to load secret keys from .env file
from dotenv import load_dotenv

# Load the .env file so we can use our OpenAI API key
load_dotenv()

# Set up the GPT-4 model
model = ChatOpenAI(
    model='gpt-4',
    temperature=1.5,           # controls randomness (higher = more creative/random)
    max_completion_tokens=10   # max length of the generated reply
)

# Send a prompt to the model
result = model.invoke("Write a 5 line poem on cricket")

# Print only the text part of the model's reply
print(result.content)

-----------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------
----------------------------------------------------------------------------
-------------------------------------------------------------------------

########################################################
## Specifying Human messages and AI messages in the prompt

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv


load_dotenv()

model = ChatOpenAI(
    model='gpt-4',
    temperature=1.5,
    max_completion_tokens=50
)

messages = [
    SystemMessage(content="You are a witty poet who only writes short, rhyming poems."),
    HumanMessage(content="Write a 5 line poem on cricket")
]

result = model.invoke(messages)
print(result.content)
