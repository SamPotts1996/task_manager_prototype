from run_model_inference import run_model_inference

class TaskPrioritizationAgent:
    def __init__(self, model_path: str, debug_mode=False):
        self.model_path = model_path
        self.debug_mode = debug_mode

    def prioritize_tasks(self, tasks_raw: str) -> list:
        """
        Sorts tasks by urgency/impact, removes duplicates, and returns a list of unique tasks.
        """
        if not tasks_raw.strip():
            if self.debug_mode:
                print("[DEBUG] No tasks for prioritization.")
            return []

        prompt = (
            "You are TaskPrioritizationAgent. Given these tasks:\n"
            f"{tasks_raw}\n\n"
            "1) Sort them by urgency and impact.\n"
            "2) Eliminate duplicates.\n"
            "3) Return the final tasks, one per line, no extra text."
        )

        response = run_model_inference(
            model_path=self.model_path,
            prompt=prompt,
            max_tokens=128,
            agent_type="TaskPrioritizationAgent",
            verbose=self.debug_mode
        )

        lines = [l.strip() for l in response.split("\n") if l.strip()]
        seen = set()
        prioritized_tasks = []
        for line in lines:
            if line not in seen:
                seen.add(line)
                prioritized_tasks.append(line)

        if not prioritized_tasks:
            prioritized_tasks = tasks_raw.splitlines()

        if self.debug_mode:
            print(f"[DEBUG] Prioritized tasks: {prioritized_tasks}")
        return prioritized_tasks
