from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence, RunnableLambda, RunnablePassthrough, RunnableParallel

load_dotenv()

# A plain Python function - counts words in a given text
# RunnableLambda will let us plug this into the chain
def word_count(text):
    return len(text.split())

# Prompt to generate a joke about a topic
prompt = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

# Set up the chat model
model = ChatOpenAI()

# Parser to extract plain text from the model's replies
parser = StrOutputParser()

# Chain to generate the joke: topic -> joke text
joke_gen_chain = RunnableSequence(prompt, model, parser)

# ---------------------------------------------------------
# parallel_chain takes the joke text and produces TWO things from it:
# 'joke'       -> the joke itself, unchanged (RunnablePassthrough)
# 'word_count' -> runs our plain Python function on the joke text
#                 (RunnableLambda wraps a normal function so it can be
#                  used as a step inside a LangChain pipeline)
# ---------------------------------------------------------
parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'word_count': RunnableLambda(word_count)
})

# Full chain: generate joke -> then run parallel_chain on that joke
final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

# Run the pipeline
# Output: {'joke': "...", 'word_count': 8}  (for example)
result = final_chain.invoke({'topic': 'AI'})

# Format joke + word count into one readable string
final_result = """{} \n word count - {}""".format(result['joke'], result['word_count'])

print(final_result)
