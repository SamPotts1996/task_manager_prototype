# main.py

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

from llama_cpp import Llama

LOG_FILE = "logs.txt"
LONG_TERM_MEMORY_FILE = "long_term_memory.txt"
SHORT_TERM_MEMORY_FILE = "short_term_memory.txt"

# A queue for user input lines from the background thread
user_input_queue = queue.Queue()

def setup_logging():
    handler = RotatingFileHandler(LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=2)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

def log_message(message: str):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")

def read_long_term_memory():
    if not os.path.exists(LONG_TERM_MEMORY_FILE):
        return ""
    with open(LONG_TERM_MEMORY_FILE, "r", encoding="utf-8") as f:
        return f.read().strip()

def append_long_term_memory(summary: str):
    summary = summary.strip()
    if summary:
        with open(LONG_TERM_MEMORY_FILE, "a", encoding="utf-8") as f:
            f.write(summary + "\n\n")

def read_short_term_memory():
    if not os.path.exists(SHORT_TERM_MEMORY_FILE):
        with open(SHORT_TERM_MEMORY_FILE, "w", encoding="utf-8"):
            pass
        return ""
    with open(SHORT_TERM_MEMORY_FILE, "r", encoding="utf-8") as f:
        return f.read()

def append_line_to_short_term_memory(line: str):
    with open(SHORT_TERM_MEMORY_FILE, "a", encoding="utf-8") as f:
        f.write(line.rstrip("\n") + "\n")

def clear_short_term_memory():
    if os.path.exists(SHORT_TERM_MEMORY_FILE):
        current = read_short_term_memory()
        if current.strip():
            log_message("=== Clearing Short-Term Memory ===")
            log_message("Content before clearing:")
            for line in current.splitlines():
                log_message(line)
            log_message("=================================\n")
    with open(SHORT_TERM_MEMORY_FILE, "w", encoding="utf-8") as f:
        f.write("")

def get_tasks():
    content = read_short_term_memory().strip()
    if not content:
        return []
    lines = content.splitlines()
    tasks = [line.strip() for line in lines if line.strip() and not line.startswith("USERINPUT#")]
    return tasks

def set_tasks(tasks_list):
    # We want to preserve existing user inputs but replace tasks
    lines = read_short_term_memory().splitlines()
    user_input_lines = [l for l in lines if l.startswith("USERINPUT#")]
    new_content = user_input_lines + tasks_list
    with open(SHORT_TERM_MEMORY_FILE, "w", encoding="utf-8") as f:
        for line in new_content:
            f.write(line + "\n")

def add_tasks(new_tasks):
    tasks = get_tasks()
    tasks.extend(new_tasks)
    set_tasks(tasks)

def pop_next_task():
    tasks = get_tasks()
    if not tasks:
        return None
    next_task = tasks.pop(0)
    set_tasks(tasks)
    return next_task

def user_input_thread():
    """
    Background thread to read user input lines. Each line is either appended
    to short-term memory or if it's 'quit'/'exit', we signal the main loop to stop.
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
    log_message(f"Starting run with objective: {user_objective}")

    clear_short_term_memory()

    # Initialize agents
    model_path = args.model_path
    task_creation_agent = TaskCreationAgent(model_path, debug_mode)
    task_prioritization_agent = TaskPrioritizationAgent(model_path, debug_mode)
    execution_agent = ExecutionAgent(model_path, debug_mode)
    long_term_memory_agent = LongTermMemoryAgent(model_path, debug_mode)
    goal_evaluation_agent = GoalEvaluationAgent(model_path, debug_mode)

    local_handler_agent = LocalHandlerAgent(model_path, debug_mode)
    external_handler_agent = ExternalHandlerAgent(model_path, debug_mode)

    init_tasks = task_creation_agent.create_tasks(user_objective, read_short_term_memory())
    set_tasks(init_tasks)
    print(Fore.MAGENTA + f"Initial Tasks: {init_tasks}")
    log_message(f"Initial Tasks: {init_tasks}")

    completed_tasks = 0
    max_iterations = 40  # safeguard

    while max_iterations > 0:
        max_iterations -= 1

        # Check user input queue
        while not user_input_queue.empty():
            line = user_input_queue.get()
            if line.lower() in ["quit", "exit"]:
                print(Fore.RED + "[Main] Stopping upon user request.")
                return
            now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            memory_line = f"USERINPUT#{now_str}#={line}"
            append_line_to_short_term_memory(memory_line)
            print(Fore.YELLOW + f"[Main] Logged user input: {memory_line}")

        tasks = get_tasks()
        if not tasks:
            print(Fore.YELLOW + "[Main] No tasks left. Attempting to create new tasks.")
            new_tasks = task_creation_agent.create_tasks(user_objective, read_short_term_memory())
            if not new_tasks:
                if goal_evaluation_agent.evaluate_progress(user_objective):
                    print(Fore.GREEN + "[Main] Objective is met. Ending run.")
                else:
                    print(Fore.YELLOW + "[Main] Objective not met, no tasks remain. Stopping.")
                break
            else:
                add_tasks(new_tasks)
                log_message(f"New tasks created: {new_tasks}")
                continue

        # Prioritize tasks
        tasks_raw = "\n".join(tasks)
        prioritized = task_prioritization_agent.prioritize_tasks(tasks_raw)
        set_tasks(prioritized)

        # Pop and execute next task
        next_task = pop_next_task()
        if not next_task:
            print(Fore.YELLOW + "[Main] No next task after prioritization.")
            continue

        print(Fore.CYAN + f"[Main] Executing: {next_task}")
        log_message(f"Executing task: {next_task}")

        try:
            if next_task.startswith("FILE#"):
                _, action, filename, *content = next_task.split("#")
                content = "#".join(content)
                if action == "create":
                    result = local_handler_agent.create_file(filename, content)
                elif action == "read":
                    result = local_handler_agent.read_file(filename)
                else:
                    result = f"Unknown file action: {action}"
            elif next_task.startswith("WEB#"):
                _, query = next_task.split("#", 1)
                result = external_handler_agent.do_web_search(query)
            else:
                result = execution_agent.execute_task(next_task)

            print(Fore.GREEN + f"[Main] Task Result: {result}")
            log_message(f"Task Result: {result}")

            summary = long_term_memory_agent.decide_what_to_store(next_task, result)
            if summary:
                append_long_term_memory(summary)
                log_message(f"Stored in LTM:\n{summary}")
            else:
                print(Fore.YELLOW + "[Main] No new insights to store.")

            completed_tasks += 1
        except Exception as e:
            err = f"[Main] Execution error on '{next_task}': {e}"
            print(Fore.RED + err)
            log_message(err)
            continue

        if goal_evaluation_agent.evaluate_progress(user_objective):
            print(Fore.GREEN + "[Main] Objective met. Ending run.")
            break

    print(Fore.GREEN + f"[Main] Done. Tasks completed: {completed_tasks}")
    log_message(f"End of run. Tasks completed: {completed_tasks}\n")

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
        log_message(msg)
        sys.exit(1)

    try:
        test_llm = Llama(model_path=args.model_path, n_gpu_layers=30, gpu_layers_size_mb=512, verbose=False)
        _ = test_llm("Test", max_tokens=1)
        del test_llm
    except Exception as e:
        msg = f"Failed to load model: {e}"
        print(Fore.RED + msg)
        log_message(msg)
        sys.exit(1)

    load_time = time.time() - start_time
    print(Fore.GREEN + f"Model loaded successfully in {load_time:.2f} seconds.")
    log_message(f"Model loaded in {load_time:.2f} seconds.")

    thread = threading.Thread(target=user_input_thread, daemon=True)
    thread.start()

    user_objective = "write hello world! to a file and save it."
    main_loop(user_objective, args.debug)

    print(Fore.GREEN + "[Main] Program ended. Goodbye.")

if __name__ == "__main__":
    main()
