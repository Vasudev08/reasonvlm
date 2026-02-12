
import subprocess
import time
import os
import requests
import sys

def check_server():
    try:
        response = requests.get("http://0.0.0.0:23333/v1/models")
        return response.status_code == 200
    except:
        return False

def main():
    model = "Qwen/Qwen2-VL-7B-Instruct"
    print(f"🚀 Starting DynaMath Official Evaluation for {model}")
    
    # 1. Start lmdeploy server
    print("⏳ Launching lmdeploy server in background...")
    # Using nohup and redirecting output to log file
    server_cmd = f"lmdeploy serve api_server {model} --server-port 23333"
    log_file = open("server.log", "w")
    process = subprocess.Popen(server_cmd, shell=True, stdout=log_file, stderr=log_file)
    
    # 2. Wait for server to be ready
    print("⏳ Waiting for server to initialize (this may take a few minutes for model loading)...")
    max_retries = 60  # 10 minutes
    retries = 0
    while not check_server():
        if retries >= max_retries:
            print("❌ Error: Server failed to start within 10 minutes. Check server.log")
            process.terminate()
            return
        
        if retries % 6 == 0:
            print(f"  ... still waiting ({retries//6}m elapsed)")
        
        time.sleep(10)
        retries += 1
    
    print("✅ Server is UP and running!")
    
    # 3. Run the evaluation script
    print("📊 Executing official DynaMath evaluation script...")
    eval_script = os.path.join("DynaMath", "evaluation", "opensource_json_eval.py")
    
    try:
        subprocess.check_call([sys.executable, eval_script])
        print("\n✅ Benchmark completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Evaluation failed with error: {e}")
    finally:
        print("🛑 Shutting down server...")
        process.terminate()

if __name__ == "__main__":
    main()
