from run_model_inference import run_model_inference

class TaskCreationAgent:
    def __init__(self, model_path: str, debug_mode=False):
        self.model_path = model_path
        self.debug_mode = debug_mode

    def create_tasks(self, objective: str, recent_tasks: str) -> list:
        """
        Generates up to 1-3 tasks relevant to 'objective', avoiding duplication from 'recent_tasks'.
        Returns a list of tasks.
        """
        prompt = (
                    f"Objective: {objective}\n\n"
                     "You are TaskCreationAgent. Provide up to 3 concise tasks required to achieve this objective. "
                        "Tasks should be independent and executable. Avoid unnecessary explanations.\n"
                            "Example:\n"
                                "- Create a file 'hello.txt' and write 'hello world!' to it."
                )


        response = run_model_inference(
            model_path=self.model_path,
            prompt=prompt,
            max_tokens=128,
            agent_type="TaskCreationAgent",
            verbose=self.debug_mode
        )

        tasks = []
        for line in response.split("\n"):
            line_clean = line.strip()
            if not line_clean:
                continue
            if line_clean.upper() == "NO TASKS REQUIRED":
                return []
            tasks.append(line_clean)

        if self.debug_mode:
            print(f"[DEBUG] Created tasks: {tasks}")
        return tasks
