from run_model_inference import run_model_inference

class ExecutionAgent:
    def __init__(self, model_path: str, debug_mode=False):
        self.model_path = model_path
        self.debug_mode = debug_mode

    def execute_task(self, task: str) -> str:
        """
        Attempts to complete 'task' or provide a short reason if not feasible.
        """
        prompt = (
            "You are ExecutionAgent. Complete the following task or provide a short reason if impossible:\n"
            f"Task: {task}\n\n"
            "Give a direct, actionable result or brief explanation if not possible."
        )

        response = run_model_inference(
            model_path=self.model_path,
            prompt=prompt,
            max_tokens=128,
            agent_type="ExecutionAgent",
            verbose=self.debug_mode
        )

        result = response.strip() or "No result."
        if self.debug_mode:
            print(f"[DEBUG] Execution result for '{task}': {result}")
        return result
