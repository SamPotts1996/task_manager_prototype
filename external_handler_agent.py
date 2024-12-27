from tools.web_tools import web_search

class ExternalHandlerAgent:
    """
    Handles external operations, like web searches or future API calls.
    """

    def __init__(self, model_path: str = "", debug_mode=False):
        self.model_path = model_path
        self.debug_mode = debug_mode

    def do_web_search(self, query: str) -> str:
        """
        Performs a web search (mock or real) with the given query.
        Returns search results or error message.
        """
        result = web_search(query)
        if self.debug_mode:
            print(f"[DEBUG] Web search for '{query}': {result}")
        return result
