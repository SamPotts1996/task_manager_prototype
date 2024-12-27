import sys
import os
import time
import datetime
import threading
import queue
import argparse
import logging
from colorama import init, Fore
from logging.handlers import RotatingFileHandler

# Agents
from agents.task_creation_agent import TaskCreationAgent
from agents.task_prioritization_agent import TaskPrioritizationAgent
from agents.execution_agent import ExecutionAgent
from agents.long_term_memory_agent import LongTermMemoryAgent
from agents.goal_evaluation_agent import GoalEvaluationAgent
from agents.local_handler_agent import LocalHandlerAgent
from agents.external_handler_agent import ExternalHandlerAgent

from tools.logs_manager import LogsManager
from task_queue import TaskQueue
from llama_cpp import Llama

# Initialize paths
LOG_FILE = "logs.txt"
TASK_LIST_FILE = "task_list.txt"

# A queue for user input lines from the background thread
user_input_queue = queue.Queue()

# Initialize logging manager and task queue
logs_manager = LogsManager(LOG_FILE)
task_queue = TaskQueue(TASK_LIST_FILE)

def setup_logging():
    handler = RotatingFileHandler(LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=2)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

def user_input_thread():
    """
    Background thread to read user input lines. Each line is either appended
    to the task queue or if it's 'quit'/'exit', we signal the main loop to stop.
    """
    while True:
        try:
            line = input().strip()
        except EOFError:
            line = ""
        if line:
            user_input_queue.put(line)

def main_loop(user_objective, debug_mode):
    print(Fore.CYAN + f"[Main] Objective: {user_objective}")
    logs_manager.log_message(f"Starting run with objective: {user_objective}")

    # Initialize agents
    model_path = args.model_path

    # The Engine
    task_creation_agent = TaskCreationAgent(model_path, debug_mode)
    task_prioritization_agent = TaskPrioritizationAgent(model_path, debug_mode)
    execution_agent = ExecutionAgent(model_path, debug_mode)
    long_term_memory_agent = LongTermMemoryAgent(model_path, debug_mode)
    goal_evaluation_agent = GoalEvaluationAgent(model_path, debug_mode)

    # The Actuators
    local_handler_agent = LocalHandlerAgent(model_path, debug_mode)
    external_handler_agent = ExternalHandlerAgent(model_path, debug_mode)

    # Load initial tasks
    init_tasks = task_creation_agent.create_tasks(user_objective, "")
    task_queue.set_tasks(init_tasks)
    print(Fore.MAGENTA + f"Initial Tasks: {init_tasks}")
    logs_manager.log_message(f"Initial Tasks: {init_tasks}")

    completed_tasks = 0
    max_iterations = 40  # safeguard

    while max_iterations > 0:
        max_iterations -= 1

        # 1) Check user_input_queue for lines
        while not user_input_queue.empty():
            line = user_input_queue.get()
            if line.lower() in ["quit", "exit"]:
                print(Fore.RED + "[Main] Stopping upon user request.")
                return
            task_queue.add_task(line)
            print(Fore.YELLOW + f"[Main] Logged user input as task: {line}")

        # 2) Check tasks
        tasks = task_queue.get_all_tasks()
        if not tasks:
            print(Fore.YELLOW + "[Main] No tasks left. Attempting to create new tasks.")
            new_tasks = task_creation_agent.create_tasks(user_objective, "")
            if not new_tasks:
                if goal_evaluation_agent.evaluate_progress(user_objective):
                    print(Fore.GREEN + "[Main] Objective is met. Ending run.")
                else:
                    print(Fore.YELLOW + "[Main] Objective not met, no tasks remain. Stopping.")
                break
            else:
                task_queue.set_tasks(new_tasks)
                logs_manager.log_message(f"New tasks created: {new_tasks}")
                continue

        # 3) Prioritize
        tasks_raw = "\n".join(tasks)
        prioritized = task_prioritization_agent.prioritize_tasks(tasks_raw)
        task_queue.set_tasks(prioritized)

        # 4) Pop & execute
        next_task = task_queue.get_next_task()
        if not next_task:
            print(Fore.YELLOW + "[Main] No next task after prioritization.")
            continue

        print(Fore.CYAN + f"[Main] Executing: {next_task}")
        logs_manager.log_message(f"Executing task: {next_task}")

        try:
            if next_task.startswith("FILE#"):
                # Handle file tasks using LocalHandlerAgent
                _, action, filename, *content = next_task.split("#")
                content = "#".join(content)  # Reconstruct content if split
                if action == "create":
                    result = local_handler_agent.create_file(filename, content)
                elif action == "read":
                    result = local_handler_agent.read_file(filename)
                else:
                    result = f"Unknown file action: {action}"
            elif next_task.startswith("WEB#"):
                # Handle web search tasks using ExternalHandlerAgent
                _, query = next_task.split("#", 1)
                result = external_handler_agent.do_web_search(query)
            else:
                # Default to ExecutionAgent for standard tasks
                result = execution_agent.execute_task(next_task)

            print(Fore.GREEN + f"[Main] Task Result: {result}")
            logs_manager.log_message(f"Task Result: {result}")

            # Store insights
            summary = long_term_memory_agent.decide_what_to_store(next_task, result)
            if summary:
                logs_manager.log_message(f"Stored in LTM:\n{summary}")
            else:
                print(Fore.YELLOW + "[Main] No new insights to store.")

            completed_tasks += 1
        except Exception as e:
            err = f"[Main] Execution error on '{next_task}': {e}"
            print(Fore.RED + err)
            logs_manager.log_message(err)
            continue

        # 5) Evaluate objective
        if goal_evaluation_agent.evaluate_progress(user_objective):
            print(Fore.GREEN + "[Main] Objective met. Ending run.")
            break

    print(Fore.GREEN + f"[Main] Done. Tasks completed: {completed_tasks}")
    logs_manager.log_message(f"End of run. Tasks completed: {completed_tasks}\n")

def main():
    init(autoreset=True)

    parser = argparse.ArgumentParser(description="Two-thread autonomous system. Main loop is fully autonomous; background thread listens for user input.")
    parser.add_argument("--model_path", required=True, help="Path to your Llama model (.gguf).")
    parser.add_argument("--debug", action="store_true", help="Enable debug prints.")
    args_parsed = parser.parse_args()

    global args
    args = args_parsed

    setup_logging()

    print(Fore.CYAN + f"Loading model from: {args.model_path}")
    start_time = time.time()
    if not os.path.exists(args.model_path):
        msg = f"Model file not found at {args.model_path}."
        print(Fore.RED + msg)
        logs_manager.log_message(msg)
        sys.exit(1)

    try:
        test_llm = Llama(model_path=args.model_path, n_gpu_layers=30, gpu_layers_size_mb=512, verbose=False)
        _ = test_llm("Test", max_tokens=1)
        del test_llm
    except Exception as e:
        msg = f"Failed to load model: {e}"
        print(Fore.RED + msg)
        logs_manager.log_message(msg)
        sys.exit(1)

    load_time = time.time() - start_time
    print(Fore.GREEN + f"Model loaded successfully in {load_time:.2f} seconds.")
    logs_manager.log_message(f"Model loaded in {load_time:.2f} seconds.")

    # Start the background thread for user input
    thread = threading.Thread(target=user_input_thread, daemon=True)
    thread.start()

    # Run main loop with a default objective
    user_objective = "understand what you are and your limitations, also how could you be improved."
    main_loop(user_objective, args.debug)

    print(Fore.GREEN + "[Main] Program ended. Goodbye.")

if __name__ == "__main__":
    main()
