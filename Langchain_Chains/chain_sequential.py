from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# 1st prompt -> ask for a detailed report on a topic
prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

# 2nd prompt -> ask for a 5-point summary of some text
prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

# Set up the chat model
model = ChatOpenAI()

# Parser to extract plain text from the model's replies
parser = StrOutputParser()

# Build a MULTI-STEP chain (this is called a "sequential chain")
# Flow: topic -> prompt1 -> model -> parser -> (becomes {text} for prompt2)
#       -> prompt2 -> model -> parser -> final summary
# Note: parser's output automatically fills prompt2's {text} placeholder,
# since the key name matches ('text' == prompt2's input_variables)
chain = prompt1 | model | parser | prompt2 | model | parser

# Run the entire pipeline with just the topic
# (report generation + summarization both happen automatically, in order)
result = chain.invoke({'topic': 'Unemployment in India'})

# Print the final 5-point summary
print(result)

# Print ASCII diagram showing the full chain structure/flow
chain.get_graph().print_ascii()
