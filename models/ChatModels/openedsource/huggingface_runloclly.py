# Import tools to run a Hugging Face model locally and chat with it
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os

# Set folder where Hugging Face models will be downloaded/stored
os.environ['HF_HOME'] = 'D:/huggingface_cache'

# Load TinyLlama model to run locally on your machine
llm = HuggingFacePipeline.from_model_id(
    model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task='text-generation',  # we want it to generate text
    pipeline_kwargs=dict(
        temperature=0.5,       # controls randomness (lower = more predictable)
        max_new_tokens=100     # max length of the generated reply
    )
)

# Wrap the model so it can chat like ChatGPT (understands roles like user/AI)
model = ChatHuggingFace(llm=llm)

# Send a question to the model
result = model.invoke("What is the capital of India")

# Print only the text part of the model's reply
print(result.content)
