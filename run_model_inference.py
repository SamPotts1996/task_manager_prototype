from llama_cpp import Llama

def run_model_inference(model_path: str, prompt: str, max_tokens=128, agent_type=None, verbose=False):
    """
    Runs inference using the specified Llama model. Adjusts parameters based on agent_type.
    Returns the text from the first choice, or an empty string if none is found.
    """
    # Default parameters
    temperature = 0.3
    top_p = 0.8
    top_k = 40

    # Adjust parameters by agent type
    if agent_type == "TaskCreationAgent":
        temperature = 0.1
        top_p = 0.7
        top_k = 20
    elif agent_type == "TaskPrioritizationAgent":
        temperature = 0.1
        top_p = 0.7
        top_k = 20
    elif agent_type == "ExecutionAgent":
        temperature = 0.15
        top_p = 0.8
        top_k = 40
    elif agent_type == "LongTermMemoryAgent":
        temperature = 0.1
        top_p = 0.75
        top_k = 20
    elif agent_type == "GoalEvaluationAgent":
        temperature = 0.0
        top_p = 0.5
        top_k = 10

    llm = Llama(
        model_path=model_path,
        n_gpu_layers=30,
        gpu_layers_size_mb=512,
        verbose=verbose,
        n_ctx=2048
    )

    stop_seq = ["\n"] if agent_type in ["GoalEvaluationAgent", "ExecutionAgent"] else None

    if verbose:
        print(f"[DEBUG] Running inference for '{agent_type or 'Unknown'}' with prompt length={len(prompt)}")

    response = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        stop=stop_seq
    )

    text = response["choices"][0]["text"].strip() if response.get("choices") else ""
    if verbose:
        print(f"[DEBUG] Model response: {text}")

    del llm
    return text
