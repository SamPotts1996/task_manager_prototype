import os

class LocalHandlerAgent:
    """
    Handles local file operations.
    """

    def __init__(self, model_path: str = "", debug_mode=False):
        self.model_path = model_path
        self.debug_mode = debug_mode

    def create_file(self, filename: str, content: str) -> str:
        """
        Creates or overwrites 'filename' with 'content'. Returns success or error message.
        """
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(content)
            msg = f"File '{filename}' created/updated successfully."
            if self.debug_mode:
                print(f"[DEBUG] {msg}")
            return msg
        except Exception as e:
            err = f"Error creating file '{filename}': {e}"
            if self.debug_mode:
                print(f"[DEBUG] {err}")
            return err

    def read_file(self, filename: str) -> str:
        """
        Reads and returns the content of 'filename', or error if missing.
        """
        if not os.path.exists(filename):
            return f"File '{filename}' does not exist."
        try:
            with open(filename, "r", encoding="utf-8") as f:
                data = f.read()
            if self.debug_mode:
                print(f"[DEBUG] Read file '{filename}', length {len(data)}.")
            return data
        except Exception as e:
            err = f"Error reading file '{filename}': {e}"
            if self.debug_mode:
                print(f"[DEBUG] {err}")
            return err
