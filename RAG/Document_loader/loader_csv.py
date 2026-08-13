# CSVLoader -> reads a CSV file and turns EACH ROW into its own Document object
from langchain_community.document_loaders import CSVLoader

# Load the CSV file 'Social_Network_Ads.csv' from your project folder
loader = CSVLoader(file_path='Social_Network_Ads.csv')

# .load() reads the CSV and returns a list of Document objects
# IMPORTANT: for CSVs, each ROW becomes its own separate Document
# (similar to how PyPDFLoader gives 1 Document per PAGE)
docs = loader.load()

# Check how many Documents we got -> this equals the number of rows in the CSV
print(len(docs))

# Print the second row (index 1) as a Document object
# page_content will show all columns of that row formatted as "column: value" pairs
# metadata will show {'source': 'Social_Network_Ads.csv', 'row': 1}
print(docs[1])
