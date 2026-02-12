
import os
import sys
import subprocess
from functools import partial

# 1. Add VLMEvalKit to path so we can import its modules
script_path = os.path.abspath(__file__)
root_dir = os.path.dirname(script_path)
vlm_kit_path = os.path.join(root_dir, "VLMEvalKit")
sys.path.insert(0, vlm_kit_path)

def setup_custom_judges():
    print("🔧 Registering custom Tamus AI models in VLMEvalKit...")
    try:
        from vlmeval.config import api_models, supported_VLM
        from vlmeval.api import GPT4V
        
        # Define the custom models
        custom_names = [
            "protected.gemini-2.0-flash-lite",
            "protected.gemini-2.5-flash-lite",
            "protected.o3-mini",
            "protected.gpt-4o-mini",
            "protected.llama-3.2",
            "protected.claude-3.5-haiku"
        ]
        
        for name in custom_names:
            # Register in api_models
            api_models[name] = partial(
                GPT4V, model=name, temperature=0, retry=10, verbose=False
            )
            # Register in supported_VLM (which is what run.py uses)
            supported_VLM[name] = api_models[name]
            
        print(f"✅ Successfully registered {len(custom_names)} custom models.")
    except ImportError as e:
        print(f"❌ Failed to register models: {e}")
        print("   Make sure VLMEvalKit is installed and in your path.")

def main():
    print("--- 🚀 Launching VLMEvalKit Wrapper ---")

    # 1. Handle API Base URL correction
    api_key = os.environ.get("OPENAI_API_KEY")
    api_base = os.environ.get("OPENAI_API_BASE", "")
    
    if "tamu.ai" in api_base and "/v1" in api_base:
        print("⚠️  Correcting TAMUS API Base URL from /v1 to /api/ ...")
        os.environ["OPENAI_API_BASE"] = api_base.replace("/v1", "/api")

    # 2. Setup the custom models in memory
    setup_custom_judges()

    # 3. Launch VLMEvalKit
    # Instead of calling main() via import (which might have side effects during import),
    # we launch it as a subprocess to keep it clean, but we pass the system path if needed.
    # Actually, to use the monkeypatched models, we MUST run in the same process.
    
    print("\nExecuting VLMEvalKit.run.main()...")
    print("---------------------------------------------------")
    
    # Import and run
    try:
        from run import main as vlm_main
        # Clear sys.argv and replace with our desired arguments
        # Defaulting to your specified model and dataset
        sys.argv = [
            "run.py",
            "--data", "DynaMath",
            "--model", "qwen2_vl_7b_instruct",
            "--judge", "protected.gemini-2.0-flash-lite",
            "--verbose"
        ]
        
        # If user passed arguments to this script, use them instead
        if len(sys.argv) > 1 and "run_vlm.py" in sys.argv[0]:
            # This is just a fallback, usually sys.argv handles it
            pass

        vlm_main()
    except Exception as e:
        print(f"\n❌ Execution failed: {e}")

if __name__ == "__main__":
    main()
