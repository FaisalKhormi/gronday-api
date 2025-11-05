import os
from dotenv import load_dotenv
from openai import OpenAI

# Load the API key from .env file
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

while True:
    user_input = input("\nEnter your question (or type 'quit' to exit): ")

    if user_input.lower() in ["quit", "exit"]:
        print("Goodbye 👋")
        break

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers in English."},
            {"role": "user", "content": user_input},
        ],
    )

    answer = response.choices[0].message.content
    print("\n🤖 GPT Answer:")
    print(answer)
