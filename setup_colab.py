
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

def apply_patches():
    print("\n🛠️ Applying Code Patches...")
    
    # Patch 1: Qwen2-VL Prompt Hijack Fix
    prompt_path = "VLMEvalKit/vlmeval/vlm/qwen2_vl/prompt.py"
    if os.path.exists(prompt_path):
        print(f"  - Patching {prompt_path}...")
        with open(prompt_path, "r") as f:
            content = f.read()
        
        old_str = "dataset not in {'MMVet', 'ChartQAPro', 'ChartQAPro_CoT', 'ChartQAPro_PoT', 'ChartMuseum'}"
        new_str = "dataset not in {'MMVet', 'ChartQAPro', 'ChartQAPro_CoT', 'ChartQAPro_PoT', 'ChartMuseum', 'DynaMath'}"
        
        if old_str in content and new_str not in content:
            content = content.replace(old_str, new_str)
            with open(prompt_path, "w") as f:
                f.write(content)
            print("    ✅ Hijack fix applied.")
        else:
            print("    ℹ️ Hijack fix already present or target string not found.")

    # Patch 2: DynaMath One-Shot Prompting
    dynamath_path = "VLMEvalKit/vlmeval/dataset/dynamath.py"
    if os.path.exists(dynamath_path):
        print(f"  - Patching {dynamath_path}...")
        with open(dynamath_path, "r") as f:
            content = f.read()
            
        # One-shot example
        one_shot_example = """    EXAMPLE = {
        "solution": "The area of a triangle is (base * height) / 2. Given base=10 and height=5, Area = (10 * 5) / 2 = 25.",
        "short answer": "25.0"
    }"""
        
        if '"solution": "[Detailed step-by-step explanation]"' in content:
            content = content.replace('    EXAMPLE = {\n        "solution": "[Detailed step-by-step explanation]",\n        "short answer": "[Concise Answer]"\n    }', one_shot_example)
            with open(dynamath_path, "w") as f:
                f.write(content)
            print("    ✅ One-Shot example applied.")
        else:
            print("    ℹ️ One-Shot example already present or target not found.")

def main():
    print("--- Setting up VLMEvalKit (Submodule) in Colab ---")

    # 1. Initialize Submodule
    print("🔗 Initializing and Updating Submodule...")
    run_command("git submodule update --init --recursive")

    # 2. Apply Patches
    apply_patches()

    # 3. Install dependencies
    print("\n📦 Installing dependencies...")
    run_command("cd VLMEvalKit && pip install -e .")
    run_command("cd VLMEvalKit && pip install -r requirements.txt")
    
    # Specific versions
    print("🔧 Installing specific transformers & flash-attn...")
    run_command("pip uninstall transformers -y")
    run_command("pip install transformers==4.47.0")
    run_command("pip install flash-attn --no-build-isolation")

    # 4. Create .env file
    env_path = os.path.join("VLMEvalKit", ".env")
    google_key = os.environ.get("GOOGLE_API_KEY", "<your GOOGLE_API_KEY>")
    openai_key = os.environ.get("OPENAI_API_KEY", "<your OPENAI_API_KEY>")
    
    if not os.path.exists(env_path):
        print("\n📝 Creating .env file...")
        env_content = f"""# API Keys for VLMEvalKit
GOOGLE_API_KEY={google_key if google_key != '<your GOOGLE_API_KEY>' else ''}
OPENAI_API_KEY={openai_key if openai_key != '<your OPENAI_API_KEY>' else ''}
"""
        with open(env_path, "w") as f:
            f.write(env_content)
        print(f"✅ Created {env_path}")

    print("\n🚀 Setup Complete!")
    print("----------------------------------------------------------------")
    print("# Run this in the next cell to set environment variables:")
    print("import os")
    print(f"os.environ['OPENAI_API_KEY'] = '{openai_key}'")
    print(f"os.environ['GOOGLE_API_KEY'] = '{google_key}'")
    print("----------------------------------------------------------------")
    print("Final run commands:")
    print("python VLMEvalKit/run.py --data DynaMath --model GPT4o_HIGH --verbose --judge gpt-4o-mini --api-nproc 1")
    print("python VLMEvalKit/run.py --data DynaMath --model Qwen2-VL-7B-Instruct --verbose --judge gpt-4o-mini")

if __name__ == "__main__":
    main()
