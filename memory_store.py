class MemoryStore:
    def __init__(self):
        self.store = {}

    def store_result(self, task: str, result: str):
        if task in self.store:
            print(f"[DEBUG] Updating existing task '{task}' with new result.")
        else:
            print(f"[DEBUG] Storing new task '{task}' with result.")
        self.store[task] = result

    def get_result(self, task: str):
        result = self.store.get(task, "Task not found.")
        print(f"[DEBUG] Retrieved result for task '{task}': {result}")
        return result

    def get_all(self):
        print("[DEBUG] Retrieving all stored tasks and results.")
        return self.store.copy()

    def clear(self):
        print("[DEBUG] Clearing all stored tasks and results.")
        self.store.clear()
