
# Colab Setup Script for VLMEvalKit (Optimized Version)
# Usage:
# 1. Copy the content of this file to a Colab cell.
# 2. Run the cell.

import os
import subprocess
import sys
import shutil

def run_command(command):
    print(f"Running: {command}")
    try:
        subprocess.check_call(command, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Warning: Command failed: {e}")

def apply_patches(base_path="."):
    print("\n🛠️ Applying Code Patches...")
    
    # Patch 1: Qwen2-VL Prompt Hijack Fix
    prompt_path = os.path.join(base_path, "vlmeval/vlm/qwen2_vl/prompt.py")
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
    dynamath_path = os.path.join(base_path, "vlmeval/dataset/dynamath.py")
    if os.path.exists(dynamath_path):
        print(f"  - Patching {dynamath_path}...")
        with open(dynamath_path, "r") as f:
            content = f.read()
            
        one_shot_example = """    EXAMPLE = {
        "solution": "The area of a triangle is (base * height) / 2. Given base=10 and height=5, Area = (10 * 5) / 2 = 25.",
        "short answer": "25.0"
    }"""
        
        if '"solution": "[Detailed step-by-step explanation]"' in content:
            content = content.replace('    EXAMPLE = {\n        "solution": "[Detailed step-by-step explanation]",\n        "short answer": "[Concise Answer]"\n    }', one_shot_example)
            with open(dynamath_path, "w") as f:
                f.write(content)
            print("    ✅ One-Shot example applied.")

def main():
    print("--- Setting up VLMEvalKit in Colab (Optimized & Flat) ---")
    print(f"📍 Current Working Directory: {os.getcwd()}")

    # 1. Handle Submodule vs Standalone
    vlm_path = "VLMEvalKit"
    if os.path.exists(vlm_path):
        print(f"📂 Found {vlm_path}. Flattening to root for easier access...")
        # Move contents of VLMEvalKit to current root
        for item in os.listdir(vlm_path):
            if item in ['.git', '.github']: continue # Skip git metadata
            s = os.path.join(vlm_path, item)
            d = os.path.join(".", item)
            if os.path.isdir(s):
                if os.path.exists(d) and not os.path.islink(d):
                    shutil.rmtree(d)
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)
        print("✅ Directory flattened.")
    else:
        print(f"🔗 {vlm_path} not found. checking if we are already in the flattened root...")
        if not os.path.exists("setup.py"):
            print("❌ Error: setup.py not found in root. Initializing via submodule...")
            run_command("git submodule update --init --recursive")
            # Flatten after init
            if os.path.exists(vlm_path):
                return main()

    # 2. Apply Patches (Now at root)
    apply_patches(".")

    # 3. Install dependencies
    print("\n📦 Installing performance backends (vLLM, LMDeploy)...")
    if os.path.exists("setup.py") or os.path.exists("pyproject.toml"):
        run_command("pip install -e .")
    else:
        print("⚠️ Warning: No setup.py found in root. Installing from VLMEvalKit subdirectory...")
        run_command("pip install -e VLMEvalKit")

    run_command("pip install vllm>=0.6.3 lmdeploy decord flash-attn --no-build-isolation")
    run_command("pip install qwen-vl-utils")

    # 4. Create .env file
    google_key = os.environ.get("GOOGLE_API_KEY", "")
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    
    print("\n📝 Creating .env file...")
    env_content = f"""# API Keys for VLMEvalKit
GOOGLE_API_KEY={google_key}
OPENAI_API_KEY={openai_key}
"""
    with open(".env", "w") as f:
        f.write(env_content)

    print("\n🚀 Setup Complete!")
    print("----------------------------------------------------------------")
    print("Recommended Run Command (Turbo Mode):")
    print("python run.py --data DynaMath --model Qwen2-VL-7B-Instruct --verbose --judge gpt-4o-mini --use-vllm --reuse")
    print("\nAlternative (LMDeploy):")
    print("python run.py --data DynaMath --model Qwen2-VL-7B-Instruct --verbose --judge gpt-4o-mini --use-lmdeploy --reuse")
    print("----------------------------------------------------------------")

if __name__ == "__main__":
    main()
