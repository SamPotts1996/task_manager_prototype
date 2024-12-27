from run_model_inference import run_model_inference

class LongTermMemoryAgent:
    def __init__(self, model_path: str, debug_mode=False):
        self.model_path = model_path
        self.debug_mode = debug_mode

    def decide_what_to_store(self, task: str, result: str) -> str:
        """
        Analyzes (task, result) for new insights. Returns a short summary or "" if none.
        """
        prompt = (
            "You are LongTermMemoryAgent. Analyze the task and result:\n"
            f"Task: {task}\n"
            f"Result: {result}\n\n"
            "Provide 1-2 bullet points of new insights or 'NO NEW INSIGHTS'."
        )

        response = run_model_inference(
            model_path=self.model_path,
            prompt=prompt,
            max_tokens=128,
            agent_type="LongTermMemoryAgent",
            verbose=self.debug_mode
        )

        lines = [l.strip() for l in response.split("\n") if l.strip()]
        insights = []
        for line in lines:
            if "NO NEW INSIGHTS" in line.upper():
                return ""
            insights.append(line)

        summary = "\n".join(insights)
        if self.debug_mode:
            print(f"[DEBUG] LTM summary: {summary}")

        return summary
