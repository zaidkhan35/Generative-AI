from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Import tools to define a structured schema (like a form the AI must fill)
from typing import TypedDict, Annotated, Optional, Literal

# ChatPromptTemplate lets us define a structured prompt with roles
# (system, human, ai) instead of one flat string like PromptTemplate
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# Set up the chat model
model = ChatOpenAI()

# Define the schema (shape) of the output we want the AI to return
class Review(TypedDict):

    key_themes: Annotated[list[str], "Write down all the key themes discussed in the review in a list"]
    summary: Annotated[str, "A brief summary of the review"]
    sentiment: Annotated[Literal["pos", "neg"], "Return sentiment of the review either negative, positive or neutral"]
    pros: Annotated[Optional[list[str]], "Write down all the pros inside a list"]
    cons: Annotated[Optional[list[str]], "Write down all the cons inside a list"]
    name: Annotated[Optional[str], "Write the name of the reviewer"]

# Wrap the model so it returns data matching the Review schema
structured_model = model.with_structured_output(Review)

# ---------------------------------------------------------
# ChatPromptTemplate: builds a prompt made of role-based messages
# instead of a single plain string
# ---------------------------------------------------------
# 'system' message -> sets the AI's behavior/persona
# 'human' message  -> the actual user input, with a {review} placeholder
prompt = ChatPromptTemplate.from_messages([
    ('system', 'You are an expert product review analyst. Extract structured information accurately from the review text provided.'),
    ('human', 'Analyze the following review:\n\n{review}')
])

# ---------------------------------------------------------
# Simple chain: prompt -> structured_model
# ---------------------------------------------------------
# prompt.invoke({'review': text}) fills the {review} placeholder
# and produces a list of formatted messages (system + human)
# those messages get passed straight into structured_model
# NOTE: we don't add StrOutputParser here, because with_structured_output
# already returns a dict matching our Review schema, not an AIMessage
chain = prompt | structured_model

review_text = """I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it's an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I'm gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.
The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.
However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung's One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.
Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful

Review by Nitish Singh
"""

# Run the chain: fills the prompt template, sends it to the model,
# and returns a dict matching the Review schema
result = chain.invoke({'review': review_text})

print(result['name'])
print(result)
