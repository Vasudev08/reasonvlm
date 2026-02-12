
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
        
        # Try to get keys from environment, or use empty strings (placeholders override real keys in VLMEvalKit)
        google_key = os.environ.get("GOOGLE_API_KEY", "")
        openai_key = os.environ.get("OPENAI_API_KEY", "")
        
        env_content = f"""# API Keys for VLMEvalKit
GOOGLE_API_KEY={google_key}
OPENAI_API_KEY={openai_key}
"""
        with open(env_path, "w") as f:
            f.write(env_content)
            
        print(f"✅ Created {env_path}")
    else:
        print("✅ .env file already exists.")

    print("\n🚀 Setup Complete!")
    print("To ensure your API keys are loaded, run this in the next cell (REPLACE WITH YOUR ACTUAL KEYS):")
    print("----------------------------------------------------------------")
    print("# Run this cell to set environment variables")
    print("import os")
    print(f"os.environ['OPENAI_API_KEY'] = '{openai_key if openai_key != '<your OPENAI_API_KEY>' else 'PASTE_YOUR_ACTUAL_OPENAI_KEY_HERE'}'")
    print(f"os.environ['GOOGLE_API_KEY'] = '{google_key if google_key != '<your GOOGLE_API_KEY>' else 'PASTE_YOUR_ACTUAL_GOOGLE_KEY_HERE'}'")
    print("----------------------------------------------------------------")
    print("Then, verify your connection:")
    print("!python verify_openai.py")
    print("----------------------------------------------------------------")
    print("Finally, run your evaluation.")
    print("\nTo match the zero-shot baseline (GPT-4o):")
    print("python VLMEvalKit/run.py --data DynaMath --model GPT4o_HIGH --verbose --judge gpt-4o-mini")
    print("\nTo run the Qwen2-VL-7B (Base):")
    print("python VLMEvalKit/run.py --data DynaMath --model Qwen2-VL-7B --verbose --judge gpt-4o-mini")
    print("\nTo run the Qwen2-VL-7B (Instruct):")
    print("python VLMEvalKit/run.py --data DynaMath --model Qwen2-VL-7B-Instruct --verbose --judge gpt-4o-mini")

if __name__ == "__main__":
    main()
