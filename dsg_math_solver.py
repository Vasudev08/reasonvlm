import os
from PIL import Image
from dotenv import load_dotenv
from dsg_utils import (
    get_symbolic_program,
    get_template_questions,
    get_dsg_response,
    get_final_question,
    get_final_dsg_response
)

# Load environment variables (for OPENAI_API_KEY)
load_dotenv()

# Define the Generic Graph DSG Program
# High-Fidelity Schema: bottom-up decomposition of graph anatomy
GENERIC_GRAPH_PROGRAM = """
p(image | concept=math-graph) =
  p(axes-limits-and-labels | concept=math-graph)
  p(origin-location-and-grid-increments | concept=math-graph, axes-limits-and-labels)
  p(curve-type-and-qualitative-shape | concept=math-graph, origin-location-and-grid-increments)
  p(precise-coordinates-of-intercepts-and-extrema | concept=math-graph, curve-type-and-qualitative-shape)
  p(image | axes-limits-and-labels, origin-location-and-grid-increments, curve-type-and-qualitative-shape, precise-coordinates-of-intercepts-and-extrema)
"""


def solve_graph_problem(image_input, question_text, answer_type="float", api_key=None, verbose=True):
    """
    Solves a math graph problem using Deep Schema Grounding with a generic DAG.
    
    Args:
        image_input: Path to image or PIL Image object.
        question_text: The math question to solve.
        answer_type: 'float', 'multiple choice', or 'free-form' (matches VLMEvalKit).
        api_key: OpenAI API key (will check environment if None).
        verbose: Print intermediate grounding steps.
    """
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError("OpenAI API Key not found. Set OPENAI_API_KEY in .env or pass it as an argument.")

    # 1. Parse the symbolic program
    if verbose:
        print("Parsing Generic Graph Program...")
    concept, final_conditioning, symbolic_program = get_symbolic_program(GENERIC_GRAPH_PROGRAM)
    
    # 2. Build templated questions
    questions = get_template_questions(concept, symbolic_program)
    
    # 3. Execute Grounding Loop
    if verbose:
        print(f"Starting Grounding Loop for Concept: {concept}")
    cache = {}
    condition_questions, condition_answers, cache, messages_to_append = get_dsg_response(
        api_key, questions, image_input, cache, verbose=verbose
    )
    
    # 4. Construct Final Grounded Context
    grounded_context = get_final_question(concept, final_conditioning, condition_answers)
    
    # 5. Get Final Result
    if verbose:
        print("\nSending final query with grounded context...")
    
    dsg_response = get_final_dsg_response(
        api_key, 
        conditioning_text=grounded_context, 
        final_question_text=question_text, 
        image_input=image_input, 
        messages_to_append=messages_to_append,
        answer_type=answer_type,
        local_vlm=vlm_model,
        verbose=verbose
    )
    
    return {
        "grounding": condition_answers,
        "context": grounded_context,
        "answer": dsg_response["answer"],
        "solution": dsg_response["solution"]
    }

if __name__ == "__main__":
    # Example usage:
    # Set your API key in .env or here
    # solve_graph_problem("path/to/graph.png", "What is the period of this function?")
    print("DSG Math Solver ready.")
    print("Generic DAG defined:")
    print(GENERIC_GRAPH_PROGRAM)
