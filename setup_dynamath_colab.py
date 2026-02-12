
# Colab Setup Script for Official DynaMath Repository
# Usage:
# 1. Copy the content of this file to a Colab cell.
# 2. Run the cell.

import os
import subprocess
import sys

def run_command(command):
    print(f"Running: {command}")
    try:
        subprocess.check_call(command, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")

def main():
    print("--- Setting up Official DynaMath Benchmark in Colab ---")

    # 1. Initialize DynaMath Submodule
    print("🔗 Initializing DynaMath submodule...")
    run_command("git submodule update --init --recursive DynaMath")

    # 2. Install dependencies for DynaMath and Model Serving
    print("📦 Installing dependencies (lmdeploy, opencv, openai, etc.)...")
    run_command("pip install lmdeploy opencv-python openai pillow numpy")
    
    # 3. Install XVFB for dynamic question generation (if user wants to generate new variants)
    print("🔧 Installing system dependencies (xvfb) for dynamic rendering...")
    run_command("sudo apt-get update && sudo apt-get install -y xvfb")

    print("\n🚀 Setup Complete!")
    print("\n--- NEXT STEPS ---")
    print("1. Set your OpenAI API Key (if you'll use it as a judge later):")
    print("   import os; os.environ['OPENAI_API_KEY'] = 'your-key-here'")
    print("\n2. Launch the Model Server:")
    print("   !lmdeploy serve api_server Qwen/Qwen2-VL-7B-Instruct --server-port 23333 &")
    print("\n3. Run the Benchmark:")
    print("   !python DynaMath/evaluation/opensource_json_eval.py")
    
if __name__ == "__main__":
    main()
