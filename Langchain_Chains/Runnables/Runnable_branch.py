from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableBranch, RunnableLambda

load_dotenv()

# Prompt to generate a detailed report on a topic
prompt1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

# Prompt to summarize a given text
prompt2 = PromptTemplate(
    template='Summarize the following text \n {text}',
    input_variables=['text']
)

# Set up the chat model
model = ChatOpenAI()

# Parser to extract plain text from the model's replies
parser = StrOutputParser()

# Chain to generate the report: topic -> detailed report text
report_gen_chain = prompt1 | model | parser

# ---------------------------------------------------------
# RunnableBranch: conditional logic (if/else) for chains
# ---------------------------------------------------------
# Takes the report text as input (x)
# Condition: if the report is LONGER than 300 words -> summarize it
# Else (report is short enough) -> RunnablePassthrough just returns it as-is,
#                                   no summarization needed
branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 300, prompt2 | model | parser),  # if condition true -> summarize
    RunnablePassthrough()  # else -> pass the report through unchanged
)

# Full chain: generate report -> THEN check length -> summarize if too long
final_chain = RunnableSequence(report_gen_chain, branch_chain)

# Run the pipeline
# If GPT's report on this topic is long, you'll get a summary
# If it's short, you'll get the original report unchanged
print(final_chain.invoke({'topic': 'Russia vs Ukraine'}))
