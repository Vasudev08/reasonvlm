
# Colab Setup Script for VLMEvalKit
# Usage:
# 1. Copy the content of this file to a Colab cell.
# 2. Run the cell.
# Note: Patches for DynaMath are already in the Vasudev08/VLMEvalKit fork

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

def main():
    print("--- Setting up VLMEvalKit in Colab ---")
    print(f"📍 Current Working Directory: {os.getcwd()}")

    # 1. Handle Submodule vs Standalone
    vlm_path = "VLMEvalKit"
    if os.path.exists(vlm_path):
        print(f"📂 Found {vlm_path}. Flattening to root for easier access...")
        # Move contents of VLMEvalKit to current root
        for item in os.listdir(vlm_path):
            if item in ['.git', '.github']: 
                continue  # Skip git metadata
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
        print(f"🔗 {vlm_path} not found. Checking if we are already in the flattened root...")
        if not os.path.exists("setup.py"):
            print("❌ Error: setup.py not found in root. Initializing via submodule...")
            run_command("git submodule update --init --recursive")
            # Re-run if submodule was initialized
            if os.path.exists(vlm_path):
                print("\n⚠️ Submodule initialized. Please re-run this script.")
                return

    # 2. Install dependencies
    print("\n📦 Installing VLMEvalKit and dependencies...")
    if os.path.exists("setup.py") or os.path.exists("pyproject.toml"):
        run_command("pip install -e .")
    else:
        print("⚠️ Warning: No setup.py found in root. Installing from VLMEvalKit subdirectory...")
        run_command("pip install -e VLMEvalKit")

    print("\n📦 Installing performance backends (vLLM, LMDeploy)...")
    run_command("pip install vllm>=0.6.3 lmdeploy decord flash-attn --no-build-isolation")
    run_command("pip install qwen-vl-utils")

    # 3. Create .env file
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
