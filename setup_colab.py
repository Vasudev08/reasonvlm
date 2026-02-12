
# Colab Setup Script for VLMEvalKit (Submodule Version)
# Usage:
# 1. Copy the content of this file to a Colab cell.
# 2. Run the cell.

import os
import subprocess
import sys

def run_command(command):
    print(f"Running: {command}")
    subprocess.check_call(command, shell=True)

def main():
    print("--- Setting up VLMEvalKit (Submodule) in Colab ---")

    # 1. Initialize Submodule
    # In Colab, you might have cloned the main repo, but submodules are often empty.
    print("🔗 Initializing and Updating Submodule...")
    run_command("git submodule update --init --recursive")

    # 2. Install dependencies
    print("📦 Installing dependencies...")
    # Install in editable mode
    run_command("cd VLMEvalKit && pip install -e .")
    # Install strict requirements
    run_command("cd VLMEvalKit && pip install -r requirements.txt")
    
    # User-requested specific versions
    print("🔧 Installing specific transformers & flash-attn...")
    run_command("pip uninstall transformers -y")
    run_command("pip install transformers==4.47.0")
    run_command("pip install flash-attn --no-build-isolation")

    # 3. Create .env file inside the submodule (it's ignored by git usually)
    env_path = os.path.join("VLMEvalKit", ".env")
    if not os.path.exists(env_path):
        print("\n📝 Creating .env file from template...")
        
        # Try to get keys from environment, or use placeholders
        tamus_key = os.environ.get("TAMUS_AI_CHAT_API_KEY", "<your TAMUS_AI_CHAT_API_KEY>")
        tamus_endpoint = os.environ.get("TAMUS_AI_CHAT_API_ENDPOINT", "https://chat-api.tamu.ai")
        
        env_content = f"""# API Keys for VLMEvalKit
# TAMUS AI Configured for OpenAI API usage in VLMEvalKit

# VLMEvalKit uses OPENAI_API_BASE and OPENAI_API_KEY
OPENAI_API_KEY={tamus_key}
OPENAI_API_BASE={tamus_endpoint}/v1/

# Original Variables for reference
TAMUS_AI_CHAT_API_KEY={tamus_key}
TAMUS_AI_CHAT_API_ENDPOINT={tamus_endpoint}

GOOGLE_API_KEY=
"""
        with open(env_path, "w") as f:
            f.write(env_content)
            
        print(f"✅ Created {env_path}")
    else:
        print("✅ .env file already exists.")

    print("\n🚀 Setup Complete!")
    print("To ensure your API keys are loaded, run this in the next cell:")
    print("----------------------------------------------------------------")
    print("import os")
    print(f"os.environ['OPENAI_API_KEY'] = '{tamus_key}'")
    print(f"os.environ['OPENAI_API_BASE'] = '{tamus_endpoint}/v1/'")
    print("----------------------------------------------------------------")
    print("Then run your evaluation:")
    # print("python VLMEvalKit/run.py --data MMBench_DEV_EN --model qwen_chat --verbose")
    # Note: User changed default model in run_vlm.py to qwen2_vl_7b_instruct and data to DynaMath
    print("python VLMEvalKit/run.py --data DynaMath --model qwen2_vl_7b_instruct --verbose")

if __name__ == "__main__":
    main()
