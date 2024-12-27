from run_model_inference import run_model_inference

class GoalEvaluationAgent:
    def __init__(self, model_path: str, debug_mode=False):
        self.model_path = model_path
        self.debug_mode = debug_mode

    def evaluate_progress(self, objective: str) -> bool:
        """
        Returns True if the given objective is deemed completed, else False.
        """
        prompt = (
            "You are GoalEvaluationAgent. Determine if this objective is met:\n"
            f"Objective: {objective}\n\n"
            "Reply 'YES' if fully met, otherwise 'NO'."
        )

        response = run_model_inference(
            model_path=self.model_path,
            prompt=prompt,
            max_tokens=16,
            agent_type="GoalEvaluationAgent",
            verbose=self.debug_mode
        )

        answer = response.strip().upper()
        if self.debug_mode:
            print(f"[DEBUG] GoalEvaluationAgent answer: {answer}")

        return answer.startswith("YES")
