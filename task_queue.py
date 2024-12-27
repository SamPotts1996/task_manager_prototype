import os

class TaskQueue:
    def __init__(self, filename="tasks.txt"):
        """
        Initializes the TaskQueue with the given file.
        If the file doesn't exist, it creates an empty one.
        """
        self.filename = filename
        if not os.path.exists(self.filename):
            with open(self.filename, "w", encoding="utf-8") as f:
                pass  # Create an empty file

    def load_tasks(self) -> list:
        """
        Loads tasks from the file and returns them as a list.
        """
        if not os.path.exists(self.filename):
            return []
        with open(self.filename, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    def save_tasks(self, tasks: list):
        """
        Saves the given list of tasks to the file.
        """
        with open(self.filename, "w", encoding="utf-8") as f:
            f.writelines(task + "\n" for task in tasks)

    def add_task(self, task: str):
        """
        Adds a new task to the file if it doesn't already exist.
        """
        tasks = self.load_tasks()
        if task not in tasks:
            tasks.append(task)
            self.save_tasks(tasks)

    def pop_next_task(self) -> str:
        """
        Removes and returns the first task from the file.
        Returns None if the task list is empty.
        """
        tasks = self.load_tasks()
        if not tasks:
            return None
        next_task = tasks.pop(0)
        self.save_tasks(tasks)
        return next_task

    def clear_tasks(self):
        """
        Clears all tasks from the file.
        """
        self.save_tasks([])

    def get_all_tasks(self) -> list:
        """
        Returns all tasks as a list.
        """
        return self.load_tasks()
