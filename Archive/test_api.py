import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()  # this reads .env and sets environment variables

key = os.getenv("OPENAI_API_KEY")

if key:
    print(f"✅ OPENAI_API_KEY is loaded. Length = {len(key)} characters.")
else:
    print("❌ OPENAI_API_KEY not found!")
