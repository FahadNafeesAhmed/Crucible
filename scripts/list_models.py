import os
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

from google import genai

client = genai.Client()
for m in client.models.list():
    print(m.name)
