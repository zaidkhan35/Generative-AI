from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate

from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Build a prompt template with one placeholder: {topic}
prompt = PromptTemplate(
    template='Generate 5 interesting facts about {topic}',
    input_variables=['topic']
)

# Set up the chat model
model = ChatOpenAI()

# Parser to extract plain text from the model's reply
parser = StrOutputParser()

# Build the chain: prompt -> model -> parser
# prompttemplate invoke is done , filled prompt goes to model, model's reply gets cleaned to plain text
chain = prompt | model | parser

# Run the chain with an actual topic
result = chain.invoke({'topic': 'cricket'})

# Print the final plain-text result
print(result)

# Print a visual diagram (ASCII art) of the chain's structure
# shows the flow: PromptInput -> PromptTemplate -> ChatOpenAI -> StrOutputParser -> Output
# useful for debugging/understanding how your pipeline is connected
chain.get_graph().print_ascii()
