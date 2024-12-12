import os

def set_environment_variables():
    os.environ["LANGCHAIN_PROJECT"] = "PREDICT"

    # Set to 'true' if you want to enable tracing
    os.environ["LANGCHAIN_TRACING_V2"] = "true"

    # Replace with the actual LangChain endpoint URL
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

    # IMPORTANT: Replace with your actual LangChain API key
    # NEVER commit your real API key to version control or share it publicly
    os.environ["LANGCHAIN_API_KEY"] = ""

    # IMPORTANT: Replace with your actual OpenAI API key
    # CRITICAL: NEVER share your OpenAI API key publicly or commit it to version control
    os.environ["OPENAI_API_KEY"] = ""

    # IMPORTANT: Replace with your actual Tavily API key
    # NEVER expose your Tavily API key in public repositories
    os.environ['TAVILY_API_KEY'] = ""