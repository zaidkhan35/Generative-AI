from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import Optional, Literal

# Import Pydantic tools instead of TypedDict
# BaseModel = base class for defining a data schema
# Field = lets you add description + default value to each field
from pydantic import BaseModel, Field

load_dotenv()




# Set up the chat model
model = ChatOpenAI()





# Define schema using Pydantic instead of TypedDict
# Pydantic actually VALIDATES the data (checks types, enforces rules)
# unlike TypedDict which is just a "hint" with no real validation


class Review(BaseModel):
    # List of key themes discussed in the review
    key_themes: list[str] = Field(description="Write down all the key themes discussed in the review in a list")

    # Short summary of the whole review
    summary: str = Field(description="A brief summary of the review")

    # Sentiment must be exactly "pos" or "neg"
    sentiment: Literal["pos", "neg"] = Field(description="Return sentiment of the review either negative, positive or neutral")

    # Optional fields with default=None (won't error if missing)
    pros: Optional[list[str]] = Field(default=None, description="Write down all the pros inside a list")
    cons: Optional[list[str]] = Field(default=None, description="Write down all the cons inside a list")
    name: Optional[str] = Field(default=None, description="Write the name of the reviewer")



# Wrap the model so it returns data matching the Review schema

structured_model = model.with_structured_output(Review)

# Send the review text and get back a structured Review object (not a plain dict)
result = structured_model.invoke("""I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.
The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.
However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.
Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful
                                 
Review by Nitish Singh
""")

# result is a Review object (not a dict)
# you can access fields with dot notation: result.name, result.sentiment, etc.
print(result)
