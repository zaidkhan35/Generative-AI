# WebBaseLoader -> scrapes/loads text content directly from a webpage URL
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Set up the chat model
model = ChatOpenAI()

# Prompt takes TWO inputs: a question, and the text to answer it from
prompt = PromptTemplate(
    template='Answer the following question \n {question} from the following text - \n {text}',
    input_variables=['question', 'text']
)

# Parser to extract plain text from the model's reply
parser = StrOutputParser()

# The webpage we want to load and read (a Flipkart product page)
url = 'https://www.flipkart.com/apple-macbook-air-m2-16-gb-256-gb-ssd-macos-sequoia-mc7x4hn-a/p/itmdc5308fa78421'

# Create the loader with our target URL
loader = WebBaseLoader(url)

# .load() fetches the page and extracts its text content into Document objects
# (usually returns just 1 Document containing the whole page's visible text)
docs = loader.load()

# Build the chain: prompt -> model -> parser
chain = prompt | model | parser

# Ask a question, using the webpage's scraped text as context
# (fixed: added the missing closing parenthesis)
print(chain.invoke({
    'question': 'What is the product that we are talking about?',
    'text': docs[0].page_content
}))
