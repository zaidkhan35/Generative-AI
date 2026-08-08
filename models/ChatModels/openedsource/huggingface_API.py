
# Import tools to connect and chat with a Hugging Face model
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# Import tool to load secret keys/tokens from .env file
from dotenv import load_dotenv

# Load the .env file so we can use our Hugging Face token
load_dotenv()

# Connect to TinyLlama model hosted on Hugging Face
llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation"  # we want it to generate text
)

# Wrap the model so it can chat like ChatGPT (understands roles like user/AI)
model = ChatHuggingFace(llm=llm)

# Send a question to the model
result = model.invoke("What is the capital of India")

# Print only the text part of the model's reply
print(result.content)
