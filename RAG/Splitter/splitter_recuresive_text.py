# RecursiveCharacterTextSplitter -> breaks long text into smaller chunks
# "Recursive" because it tries multiple separators (paragraphs, then sentences,
# then words) to split as cleanly as possible, instead of just cutting mid-sentence
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Sample long text about space exploration
text = """
Space exploration has led to incredible scientific discoveries. From landing on the Moon to exploring Mars, humanity continues to push the boundaries of what's possible beyond our planet.
These missions have not only expanded our knowledge of the universe but have also contributed to advancements in technology here on Earth. Satellite communications, GPS, and even certain medical imaging techniques trace their roots back to innovations driven by space programs.
"""

# Set up the splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,     # max number of characters per chunk
    chunk_overlap=0,    # how many characters repeat between consecutive chunks (0 = no overlap)
)

# Split the text into chunks based on the settings above
chunks = splitter.split_text(text)

# Check how many chunks the text got split into
print(len(chunks))

# Print all the chunks to see how the text was divided
print(chunks)
