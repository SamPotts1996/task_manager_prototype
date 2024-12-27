import os

try:
    from llama_cpp import Llama
except ModuleNotFoundError:
    print("The module 'llama_cpp' is not installed or unavailable in this environment.")
    print("Please ensure it is installed using pip and accessible in your runtime.")
    exit(1)

# Define the model path from your project
MODEL_PATH = r"C:\Users\Life\Desktop\Testing\model_test\model\Llama-3.2-3B-Instruct-Q8_0.gguf"

def test_model():
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

        print("Loading model...")
        llm = Llama(model_path=MODEL_PATH, n_gpu_layers=30, gpu_layers_size_mb=512, verbose=True, device="cuda:1")

        # Test the model with a simple prompt
        prompt = "What is the capital of France?"
        print(f"Running inference on prompt: {prompt}")
        
        response = llm(prompt, max_tokens=64, temperature=0.7, top_p=0.9, top_k=40)
        result = response["choices"][0]["text"].strip() if response["choices"] else "No result."

        print(f"Model response: {result}")

    except FileNotFoundError as fe:
        print(f"File error: {fe}")
    except Exception as e:
        print(f"An error occurred while testing the model: {e}")

if __name__ == "__main__":
    test_model()
