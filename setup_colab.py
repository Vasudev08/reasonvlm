
# Colab Setup Script for VLMEvalKit (Simplified)
# Usage:
# 1. Copy the content of this file to a Colab cell.
# 2. Run the cell.
# Note: Patches for DynaMath are already in the Vasudev08/VLMEvalKit fork

import os
import subprocess

def run_command(command):
    """Run a shell command and handle errors gracefully."""
    print(f"Running: {command}")
    try:
        subprocess.check_call(command, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Warning: Command failed: {e}")

def main():
    print("=" * 60)
    print("Setting up VLMEvalKit in Colab")
    print("=" * 60)
    print(f"\n📍 Current Working Directory: {os.getcwd()}\n")

    # 1. Check if VLMEvalKit exists as subdirectory or if we're already in it
    if os.path.exists("VLMEvalKit"):
        print("✅ Found VLMEvalKit subdirectory")
        base_path = "VLMEvalKit"
    elif os.path.exists("vlmeval"):
        print("✅ Already in VLMEvalKit directory")
        base_path = "."
    else:
        print("❌ Error: VLMEvalKit not found. Please clone the repository first:")
        print("   git clone https://github.com/Vasudev08/VLMEvalKit.git")
        return

    # 2. Install VLMEvalKit
    print("\n📦 Installing VLMEvalKit...")
    if os.path.exists(os.path.join(base_path, "setup.py")):
        run_command(f"pip install -e {base_path}")
    else:
        print("⚠️ Warning: setup.py not found, trying editable install anyway...")
        run_command(f"pip install -e {base_path}")

    # 3. Optional: Install Flash Attention (may fail on some systems)
    print("\n📦 Installing Flash Attention (optional, may fail)...")
    run_command("pip install flash-attn --no-build-isolation || echo 'Flash Attention install failed, continuing anyway'")


    # 4. Create .env file
    google_key = os.environ.get("GOOGLE_API_KEY", "")
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    
    if google_key or openai_key:
        print("\n📝 Creating .env file with API keys...")
        env_path = os.path.join(base_path, ".env") if base_path != "." else ".env"
        env_content = f"""# API Keys for VLMEvalKit
GOOGLE_API_KEY={google_key}
OPENAI_API_KEY={openai_key}
"""
        with open(env_path, "w") as f:
            f.write(env_content)
        print(f"   ✅ .env file created at: {env_path}")
    else:
        print("\n⚠️ No API keys found in environment")
        print("   Set GOOGLE_API_KEY and/or OPENAI_API_KEY before running")

    # 5. Print success message and usage instructions
    print("\n" + "=" * 60)
    print("🚀 Setup Complete!")
    print("=" * 60)
    
    if base_path != ".":
        print(f"\n💡 Change to VLMEvalKit directory:")
        print(f"   %cd {base_path}")
    
    print("\n📋 Example Run Commands:")
    print("\n   Turbo Mode (vLLM):")
    print("   python run.py --data DynaMath --model Qwen2-VL-7B-Instruct \\")
    print("                 --verbose --judge gpt-4o-mini --use-vllm --reuse")
    
    print("\n   Alternative (LMDeploy):")
    print("   python run.py --data DynaMath --model Qwen2-VL-7B-Instruct \\")
    print("                 --verbose --judge gpt-4o-mini --use-lmdeploy --reuse")
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
