from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# RunnablePassthrough -> passes its input through UNCHANGED (does nothing to it)
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough

load_dotenv()

# Prompt to generate a joke about a topic
prompt1 = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

# Set up the chat model
model = ChatOpenAI()

# Parser to extract plain text from the model's replies
parser = StrOutputParser()

# Prompt to explain a given joke
prompt2 = PromptTemplate(
    template='Explain the following joke - {text}',
    input_variables=['text']
)

# Chain to generate the joke: topic -> joke text
joke_gen_chain = RunnableSequence(prompt1, model, parser)

# ---------------------------------------------------------
# RunnablePassthrough explained:
# ---------------------------------------------------------
# The output of joke_gen_chain (the joke text) needs to be used TWICE:
# 1. As the final 'joke' to show the user (unchanged, as-is)
# 2. As input to prompt2, to generate an 'explanation' of that joke
#
# RunnablePassthrough() just means "take whatever came in, and pass it out
# exactly as it is, with no changes" -> this is how we keep a copy of the
# original joke text alongside the explanation
parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),                       # keeps the joke unchanged
    'explanation': RunnableSequence(prompt2, model, parser)  # generates explanation from the joke
})

# Full chain: first generate the joke, THEN run parallel_chain on that joke
# parallel_chain receives the joke text as input, and produces both
# 'joke' (same text) and 'explanation' (new text) from it
final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

# Run the pipeline with a topic
# Output: {'joke': "...", 'explanation': "..."}
print(final_chain.invoke({'topic': 'cricket'}))
