from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# RunnableParallel -> runs multiple chains AT THE SAME TIME (in parallel)
# instead of one after another (sequentially)
from langchain.schema.runnable import RunnableParallel

load_dotenv()

# Two different models from two different companies
model1 = ChatOpenAI()
model2 = ChatAnthropic(model_name='claude-3-7-sonnet-20250219')

# Prompt to generate short notes from text
prompt1 = PromptTemplate(
    template='Generate short and simple notes from the following text \n {text}',
    input_variables=['text']
)

# Prompt to generate quiz questions from text
prompt2 = PromptTemplate(
    template='Generate 5 short question answers from the following text \n {text}',
    input_variables=['text']
)

# Prompt to merge the notes + quiz into one final document
prompt3 = PromptTemplate(
    template='Merge the provided notes and quiz into a single document \n notes -> {notes} and quiz -> {quiz}',
    input_variables=['notes', 'quiz']
)

# Parser to extract plain text from model replies
parser = StrOutputParser()

# ---------------------------------------------------------
# RunnableParallel: run 2 independent chains AT THE SAME TIME
# ---------------------------------------------------------
# 'notes' chain -> uses OpenAI to generate notes
# 'quiz' chain  -> uses Claude to generate quiz questions
# Both chains get the SAME input {text}, but run simultaneously
# (faster than running one after the other, since they don't depend on each other)
# Output is a dict: {'notes': "...", 'quiz': "..."}
parallel_chain = RunnableParallel({
    'notes': prompt1 | model1 | parser,
    'quiz': prompt2 | model2 | parser
})

# Chain to merge the parallel outputs (notes + quiz) into one final document
merge_chain = prompt3 | model1 | parser

# Full chain: first run notes+quiz in parallel, THEN merge them together
# Output of parallel_chain (a dict with 'notes' and 'quiz' keys)
# automatically fills prompt3's {notes} and {quiz} placeholders
chain = parallel_chain | merge_chain

# Sample text to process (about Support Vector Machines)
text = """
Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection.
The advantages of support vector machines are:
Effective in high dimensional spaces.
Still effective in cases where number of dimensions is greater than the number of samples.
Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.
Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.
The disadvantages of support vector machines include:
If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial.
SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation (see Scores and probabilities, below).
The support vector machines in scikit-learn support both dense (numpy.ndarray and convertible to that by numpy.asarray) and sparse (any scipy.sparse) sample vectors as input. However, to use an SVM to make predictions for sparse data, it must have been fit on such data. For optimal performance, use C-ordered numpy.ndarray (dense) or scipy.sparse.csr_matrix (sparse) with dtype=float64.
"""

# Run the full pipeline: parallel generation -> merge into final document
result = chain.invoke({'text': text})

# Print the final merged document
print(result)

# Print ASCII diagram showing the chain's structure
# This will show a BRANCHING structure (splits into 2 paths, then joins back into 1)
chain.get_graph().print_ascii()
