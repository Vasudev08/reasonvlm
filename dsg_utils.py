import base64
import re
import requests
import json
import io
from PIL import Image
from openai import OpenAI

def get_symbolic_program(program_str):
    """
    Extract the concept, final conditioning, and conditional programs (term, conditions) from a symbolic program string.

    Args:
        program_str (str): A string containing the symbolic program.

    Returns:
        tuple:
            - concept (str): The primary concept from the program.
            - final_conditioning (list): A list of final conditions extracted from the program.
            - cond_programs (list of tuples): A list of (term, conditions) pairs.
    """
    concept_pattern = r'[ \t]*image[ \t]*\|[ \t]*concept[ \t]*=[ \t]*(.*)[ \t]*\)[ \t]*='
    term_pattern = r'[ \t]*(.*)[ \t]*\|[ \t]*'
    final_pattern = r'[ \t]*.*[ \t]*\|[ \t]*(.*)[ \t]*\)'

    # Remove newlines and split on "p(" tokens
    segments = program_str.replace('\n', '').split('p(')

    concept = None
    final_conditioning = []
    cond_programs = []

    for idx, segment in enumerate(segments):
        s = segment.strip()
        if not s:
            continue

        # Identify concept
        if s.endswith('='):
            match_concept = re.search(concept_pattern, s)
            if match_concept:
                concept = match_concept.group(1).strip()

        # Identify final conditioning
        elif s.startswith('image') and (idx + 1 == len(segments)):
            match_final = re.search(final_pattern, s)
            if match_final:
                final_str = match_final.group(1)
                final_conditioning = [c.replace('-', ' ').strip() for c in final_str.split(',')]

        # Otherwise, extract terms and conditions
        else:
            match_term = re.search(term_pattern, s)
            if match_term:
                term = match_term.group(1).replace('-', ' ').strip()
                # The part after the last '|'
                raw_conditions = s.split('|')[-1].split(',')
                # Filter out anything containing the concept, then clean up
                cs = [
                    c.replace(')', '').replace('-', ' ').strip()
                    for c in raw_conditions
                    if concept not in c
                ]
                cond_programs.append((term, cs))

    return concept, final_conditioning, cond_programs


def get_template_questions(concept, symbolic_program):
    """
    Build templated questions for each (term, conditions) entry from the symbolic program.
    """
    questions = []
    question_fmt = "Imagine that the image represents {concept}{conditions} what is the {term}?"
    condition_fmt = " and the {c} is {{{c}}},"

    for term, conds in symbolic_program:
        if not conds:
            # No conditioning
            question_str = question_fmt.format(concept=concept, conditions=",", term=term)
            question_str = question_str.replace(",,", ",")
        else:
            # Build the " and the {c} is {val}," parts
            conditions_str = "".join(condition_fmt.format(c=c) for c in conds)
            question_str = question_fmt.format(concept=concept, conditions=conditions_str, term=term)

        questions.append((term, question_str))
    return questions


def get_conditional_question(question, condition_answers):
    """
    Replace placeholders of the form {key} in a question with values from condition_answers.
    """
    placeholders = re.findall(r'\{(.*?)\}', question)
    for ph in placeholders:
        question = question.replace(f'{{{ph}}}', str(condition_answers.get(ph, '')))
    return question


def get_final_question(concept, final_conditions, condition_answers):
    """
    Construct a final question based on the main concept and final conditions.
    """
    base_fmt = "Imagine that the image represents {concept}{conditions}."
    cond_fmt = " and the {c} is {val}"

    # Build the " and the {c} is {val}" parts
    conditions_str = "".join(
        cond_fmt.format(c=c, val=condition_answers.get(c, ''))
        for c in final_conditions
    )
    return base_fmt.format(concept=concept, conditions=conditions_str)


def encode_image(image_input):
    """
    Encode an image (path or PIL Image) to Base64.
    """
    if isinstance(image_input, str):
        with open(image_input, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    elif isinstance(image_input, Image.Image):
        buffered = io.BytesIO()
        image_input.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    else:
        raise ValueError("Unsupported image input type. Provide path string or PIL Image.")


def get_dsg_response(api_key, questions, image_input, cache, verbose=True):
    """
    Send each question with the image to GPT-4o, caching responses.
    """
    test_image_b64 = encode_image(image_input)
    condition_answers = {}
    condition_questions = {}
    messages_to_append = []

    strict_query_suffix = " Answer with one word or phrase only, do not elaborate."
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    for term, question_template in questions:
        resolved_question = get_conditional_question(question_template, condition_answers)
        resolved_question += strict_query_suffix
        condition_questions[term] = resolved_question

        if verbose:
            print(f"Querying [{term}]: {resolved_question}")

        # Unique cache key (using the question text)
        cache_key = resolved_question
        if cache_key in cache:
            response = cache[cache_key]
        else:
            payload = {
                "model": "gpt-4o",
                "messages": [
                    {
                        'role': 'system',
                        'content': "You are an accurate question answering engine that grounds abstract concepts into visual objects. Disregard grammar issues."
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": resolved_question},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{test_image_b64}"}},
                        ]
                    }
                ],
                "max_tokens": 100,
                "temperature": 0.0
            }

            resp = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            if resp.status_code != 200:
                print(f"Error: {resp.text}")
                response = "error"
            else:
                resp_json = resp.json()
                response = resp_json['choices'][0]['message']['content'].lower().strip().replace('.', '')

            # Build user & assistant messages to store for final context
            user_message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": resolved_question},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{test_image_b64}"}},
                ]
            }
            assistant_message = {
                "role": "assistant",
                "content": [{"type": "text", "text": response}]
            }
            messages_to_append.extend([user_message, assistant_message])
            cache[cache_key] = response

        if verbose:
            print(f"Answer: {response}")
        condition_answers[term] = response

    return condition_questions, condition_answers, cache, messages_to_append



# --- VLMEvalKit Style Prompting Constants ---
VLMEVAL_GUIDE = """
to the following JSON format, which includes two keys: 'solution' and 'short answer'. The 'solution' key can contain \
detailed steps needed to solve the question, and the 'short answer' key should provide a concise response. {INST}

Example of expected JSON response format:
"""

VLMEVAL_EXAMPLE = {
    "solution": "[Detailed step-by-step explanation]",
    "short answer": "[Concise Answer]"
}

def parse_dsg_json(response_text):
    """
    Attempts to parse the JSON response from the VLM to extract the 'short answer'.
    """
    try:
        # Find the JSON block in the response
        if "```json" in response_text:
            json_str = response_text.split("```json")[1].split("```")[0].strip()
        elif "{" in response_text:
            json_str = response_text[response_text.find("{"): response_text.rfind("}") + 1]
        else:
            json_str = response_text.strip()
            
        data = json.loads(json_str)
        return data.get("short answer", response_text), data.get("solution", "")
    except Exception:
        # Fallback: just return the raw text if parsing fails
        return response_text, ""

def get_final_dsg_response(api_key, conditioning_text, final_question_text, image_input, messages_to_append, answer_type="float", local_vlm=None, verbose=True):
    """
    Final answer query incorporating the grounded context and VLMEvalKit formatting.
    If local_vlm is provided (a VLMEvalKit model instance), it uses that for inference.
    """
    """
    Final answer query incorporating the grounded context and VLMEvalKit formatting.
    """
    test_image_b64 = encode_image(image_input)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    # Determine standard instruction based on answer_type (matching VLMEvalKit)
    if answer_type == 'multiple choice':
        inst = "Provide the corresponing choice option in the 'short answer' key, such as 'A', 'B', 'C', or 'D'."
    elif answer_type == 'float' or answer_type == 'numeric':
        inst = "Format the answer as a three-digit floating-point number and provide it in the 'short answer' key."
    else:
        inst = "Float numbers in the answer should be formatted as three-digit floating-point numbers."

    # Construct the VLMEvalKit style prompt
    structured_prompt = (
        f"## Context\n{conditioning_text}\n\n"
        f"## Question\n{final_question_text}\n\n"
        f"Please solve the graph problem based on the provided image and context. Use the following JSON format: "
        f"{{'solution': '[Detailed step-by-step reasoning]', 'short answer': '[Concise Answer]'}}. "
        f"{inst}\n"
    )

    if local_vlm is not None:
        if verbose:
            print(f"--- Running Local Inference using {type(local_vlm).__name__} ---")
        
        # VLMEvalKit models expect list of [image_path, prompt] or similar
        # Since Image is already loaded, we ensure it handles it correctly.
        # Most VLMEvalKit models take a list of message/components
        
        # We need a temp file for VLMEvalKit generate if it doesn't take PIL directly
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            if isinstance(image_input, Image.Image):
                image_input.save(tmp.name)
            else:
                tmp.write(open(image_input, "rb").read())
            tmp_path = tmp.name

        try:
            # Standard VLMEvalKit generate call
            raw_response = local_vlm.generate([tmp_path, structured_prompt])
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    else:
        # OpenAI API Fallback (GPT-4o)
    short_answer, solution = parse_dsg_json(raw_response)
    
    if verbose:
        print(f"\n--- DSG SOLUTION ---\n{solution}\n")
        print(f"Final DSG Answer: {short_answer}\n")

    return {"answer": short_answer, "solution": solution}
