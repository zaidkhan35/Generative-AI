# SemanticChunker -> splits text based on MEANING, not character count
# It uses embeddings to detect where the topic actually shifts
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Set up the semantic splitter
# It uses OpenAI embeddings under the hood to measure how "similar in meaning"
# consecutive sentences are
text_splitter = SemanticChunker(
    OpenAIEmbeddings(),

    # "standard_deviation" -> decide a split point based on how much the
    # similarity between sentences varies statistically
    breakpoint_threshold_type="standard_deviation",

    # how sensitive the splitter is - higher number = fewer, bigger chunks
    # (only splits on VERY strong topic changes)
    breakpoint_threshold_amount=3
)

# Sample text with 3 clearly different topics mixed together:
# farming, cricket/IPL, and terrorism
sample = """
Farmers were working hard in the fields, preparing the soil and planting seeds for the next season. The sun was bright, and the air smelled of earth and fresh grass. The Indian Premier League (IPL) is the biggest cricket league in the world. People all over the world watch the matches and cheer for their favourite teams.
Terrorism is a big danger to peace and safety. It causes harm to people and creates fear in cities and villages. When such attacks happen, they leave behind pain and sadness. To fight terrorism, we need strong laws, alert security forces, and support from people who care about peace and safety.
"""

# create_documents() -> embeds sentences, measures meaning-shifts between them,
# and splits into chunks wherever the topic clearly changes
docs = text_splitter.create_documents([sample])

# Check how many semantic chunks were created
# ideally close to 3, since there are 3 distinct topics in the sample text
print(len(docs))

# Print the actual chunks - each one should roughly correspond to one topic
print(docs)
