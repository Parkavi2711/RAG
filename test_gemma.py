from langchain_community.llms import Ollama

# Connect to local Ollama
llm = Ollama(model="gemma3:1b")

# Ask a simple question
response = llm.invoke("Explain blockchain in 3 lines.")

print("\nResponse from Gemma:\n")
print(response)