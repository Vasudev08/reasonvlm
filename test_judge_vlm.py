
import os
import sys
from functools import partial

# 1. Setup paths
script_path = os.path.abspath(__file__)
root_dir = os.path.dirname(script_path)
vlm_kit_path = os.path.join(root_dir, "VLMEvalKit")
sys.path.insert(0, vlm_kit_path)

def test_judge():
    print("--- 🕵️ Verifying Judge Integration ---")
    
    # 2. Correct URL if needed
    api_base = os.environ.get("OPENAI_API_BASE", "")
    if "tamu.ai" in api_base and "/v1" in api_base:
        print("⚠️  Correcting TAMUS API Base URL...")
        os.environ["OPENAI_API_BASE"] = api_base.replace("/v1", "/api")

    # 3. Register custom models in VLMEvalKit
    try:
        from vlmeval.config import api_models, supported_VLM
        from vlmeval.api import GPT4V
        
        model_name = "protected.gemini-2.0-flash-lite"
        print(f"Registering {model_name}...")
        
        api_models[model_name] = partial(
            GPT4V, model=model_name, temperature=0, retry=10, verbose=True
        )
        supported_VLM[model_name] = api_models[model_name]
        
        # 4. Try to instantiate and call the model
        print(f"Instantiating {model_name} as a judge...")
        judge_model = supported_VLM[model_name]()
        
        test_prompt = "Hello! Please answer this sample query for a test: What is 2+2? Answer only with the number."
        print(f"Sending test prompt to judge...")
        
        # GPT4V wrapper usually has a 'generate' method
        response = judge_model.generate(test_prompt)
        
        print("\n---------------------------------------------------")
        if response and "4" in response:
            print(f"✅ INTEGRATION SUCCESS!")
            print(f"Judge Responded: {response}")
            print("The VLMEvalKit judge system is correctly using your TAMUS API.")
        else:
            print(f"⚠️  Incomplete Response.")
            print(f"Judge Responded: {response}")
            print("The connection worked, but the response was unexpected.")
            
    except Exception as e:
        print(f"\n❌ Integration Failed!")
        print(f"Error: {e}")
        print("\nCheck if VLMEvalKit is properly set up and dependencies are installed.")

if __name__ == "__main__":
    test_judge()
