import os
from data_utils import download_dynamath, get_item_data
from dsg_math_solver import solve_graph_problem
from PIL import Image

def test_viser_implementation():
    # 1. Download sample data
    dataset = download_dynamath(split='sample_variant1')
    if not dataset:
        print("Failed to download dataset.")
        return
    
    # 2. Pick a sample item (e.g., the first one)
    item = dataset[0]
    question_text, img, answer_type, correct_answer = get_item_data(item)
    
    print(f"\n--- Testing with Question: {question_text} ---")
    print(f"Correct Answer: {correct_answer}")
    
    # 3. Solve WITHOUT VISER
    print("\n>>> Running WITHOUT VISER...")
    try:
        res_no_viser = solve_graph_problem(img, question_text, answer_type=answer_type, use_viser=False)
        print(f"Result (No VISER): {res_no_viser['answer']}")
    except Exception as e:
        print(f"Error without VISER: {e}")
        
    # 4. Solve WITH VISER
    print("\n>>> Running WITH VISER...")
    try:
        res_viser = solve_graph_problem(img, question_text, answer_type=answer_type, use_viser=True)
        print(f"Result (With VISER): {res_viser['answer']}")
    except Exception as e:
        print(f"Error with VISER: {e}")

if __name__ == "__main__":
    # Ensure OPENAI_API_KEY is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set your OPENAI_API_KEY environment variable or in a .env file.")
    else:
        test_viser_implementation()
