from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# RunnableSequence -> same idea as using "|" to chain steps, just written explicitly
# RunnableParallel -> runs multiple chains AT THE SAME TIME
from langchain.schema.runnable import RunnableSequence, RunnableParallel

load_dotenv()

# Prompt to generate a tweet about a topic
prompt1 = PromptTemplate(
    template='Generate a tweet about {topic}',
    input_variables=['topic']
)

# Prompt to generate a LinkedIn post about the same topic
prompt2 = PromptTemplate(
    template='Generate a Linkedin post about {topic}',
    input_variables=['topic']
)

# Set up the chat model
model = ChatOpenAI()

# Parser to extract plain text from the model's replies
parser = StrOutputParser()

# ---------------------------------------------------------
# RunnableSequence(prompt1, model, parser) is the SAME thing as writing:
#   prompt1 | model | parser
# Just written out explicitly using the RunnableSequence class
# instead of the shorthand "|" pipe operator
# ---------------------------------------------------------
parallel_chain = RunnableParallel({
    # 'tweet' chain -> generates a tweet
    'tweet': RunnableSequence(prompt1, model, parser),

    # 'linkedin' chain -> generates a LinkedIn post
    # Both chains use the SAME input {topic}, but run simultaneously
    'linkedin': RunnableSequence(prompt2, model, parser)
})

# Run both chains at once with the same topic
# Output is a dict: {'tweet': "...", 'linkedin': "..."}
result = parallel_chain.invoke({'topic': 'AI'})

# Print each result separately
print(result['tweet'])
print(result['linkedin'])
