import cohere

# Replace with your actual Cohere API key
api_key = ""

# Replace with the path to your code file
code_file_path = r"C:\move\Code\VSCode\Python\DAA2year1\Week 4\Huffman Coding.py"

# Initialize the Cohere API client
co = cohere.Client(api_key)

# Read the content of the code file
with open(code_file_path, "r") as file:
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
