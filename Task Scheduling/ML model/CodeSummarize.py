import os
import sys

import cohere

# The old key that was hardcoded here is burned — it was committed to git.
# Revoke it at dashboard.cohere.com and export a new one instead.
api_key = os.environ.get("COHERE_API_KEY")
if not api_key:
    sys.exit("Set COHERE_API_KEY in your environment.")

if len(sys.argv) < 2:
    sys.exit("Usage: python CodeSummarize.py <path-to-code-file>")
code_file_path = sys.argv[1]

# Initialize the Cohere API client
co = cohere.Client(api_key)

# Read the content of the code file
with open(code_file_path, "r", encoding="utf-8") as file:
    code_content = file.read()

# Generate a description for the code file
response = co.generate(
    model="command-nightly",
    prompt=f"Generate a concise description of the functionality and key features of the following code: {code_content}",
    max_tokens=200,  # Adjust as needed
)

# Print the generated description
print("Generated Description for Code:")
print(response.generations[0].text)
