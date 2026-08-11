from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# Set up the chat model
model = ChatOpenAI()

# Define schema as a plain JSON Schema dictionary
# (no TypedDict or Pydantic class needed — just raw dict/JSON)
# Useful when the schema comes from outside Python (e.g. an API, a config file,
# or a language-agnostic source) or you don't want a Python class at all
json_schema = {
  "title": "Review",
  "type": "object",
  "properties": {
    # List of key themes discussed in the review
    "key_themes": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "Write down all the key themes discussed in the review in a list"
    },
    # Short summary of the review
    "summary": {
      "type": "string",
      "description": "A brief summary of the review"
    },
    # Sentiment must be exactly "pos" or "neg" (enum restricts allowed values)
    "sentiment": {
      "type": "string",
      "enum": ["pos", "neg"],
      "description": "Return sentiment of the review either negative, positive or neutral"
    },
    # Optional field — can be a list of strings or null
    "pros": {
      "type": ["array", "null"],
      "items": {
        "type": "string"
      },
      "description": "Write down all the pros inside a list"
    },
    "cons": {
      "type": ["array", "null"],
      "items": {
        "type": "string"
      },
      "description": "Write down all the cons inside a list"
    },
    # Optional field — string or null
    "name": {
      "type": ["string", "null"],
      "description": "Write the name of the reviewer"
    }
  },
  # These fields MUST be present in the output (others are optional)
  "required": ["key_themes", "summary", "sentiment"]
}

# Wrap the model so it returns data matching the JSON schema
structured_model = model.with_structured_output(json_schema)

# Send the review text and get back structured data (usually a plain dict)
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

# result is usually a plain dictionary matching the JSON schema
print(result)
