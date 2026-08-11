
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# Build the template
chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful {domain} expert'),
    ('human', 'Explain in simple terms, what is {topic}')
])

# Fill in the placeholders
prompt = chat_template.invoke({'domain': 'cricket', 'topic': 'Dusra'})

# Set up the model
model = ChatOpenAI()

# Send the filled-in prompt to the model
result = model.invoke(prompt)

# Print the AI's actual answer
print(result.content)
