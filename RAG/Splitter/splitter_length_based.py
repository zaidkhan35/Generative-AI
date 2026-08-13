# CharacterTextSplitter -> simpler splitter than RecursiveCharacterTextSplitter
# It splits using ONE fixed separator only (no fallback strategy)
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Load the PDF - each page becomes its own Document object
loader = PyPDFLoader('dl-curriculum.pdf')
docs = loader.load()

# Set up the splitter
splitter = CharacterTextSplitter(
    chunk_size=200,      # max number of characters per chunk
    chunk_overlap=0,     # no repeated characters between chunks
    separator=''         # split purely by character count, ignore words/sentences entirely
)

# split_documents() -> works directly on Document objects (not plain text)
# It splits each Document's page_content into smaller chunks,
# and creates a NEW Document for each chunk (keeping the original metadata)
result = splitter.split_documents(docs)

# Print the content of the second chunk (index 1)
print(result[1].page_content)
