# PyPDFLoader -> reads a PDF file and turns each page into a Document object
from langchain_community.document_loaders import PyPDFLoader

# Load the PDF file 'dl-curriculum.pdf' from your project folder
loader = PyPDFLoader('dl-curriculum.pdf')

# .load() reads the PDF and returns a list of Document objects
# IMPORTANT: for PDFs, each PAGE becomes its own separate Document
# (unlike TextLoader, which usually gives you just 1 Document for the whole file)
docs = loader.load()

# Check how many Documents we got -> this equals the number of pages in the PDF
print(len(docs))

# Print the text content of the FIRST page (index 0)
print(docs[0].page_content)

# Print metadata of the SECOND page (index 1)
# e.g. {'source': 'dl-curriculum.pdf', 'page': 1}
# metadata tells you which file and which page this Document came from
print(docs[1].metadata)
