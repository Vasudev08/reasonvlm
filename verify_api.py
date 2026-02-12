
import os
import requests
import sys
import json

def verify_api():
    print("--- 🕵️ Verifying API Configuration (Requests Method) ---")
    
    # 1. Check Environment Variables
    # The setup script sets OPENAI_API_BASE to "{endpoint}/api/"
    # The user manual curl uses "{endpoint}/api/chat/completions"
    
    api_key = os.environ.get("OPENAI_API_KEY")
    api_base = os.environ.get("OPENAI_API_BASE")
    
    print(f"Checking Environment Variables:")
    print(f"  OPENAI_API_KEY: {'✅ Set' if api_key else '❌ Not Set'}")
    print(f"  OPENAI_API_BASE: {api_base if api_base else '❌ Not Set'}")
    
    if not api_key or not api_base:
        print("\n❌ Error: Missing API Key or Base URL.")
        print("Please run the setup code from the previous step.")
        return

    # Force Tamus AI Base URL correction if needed
    if "tamu.ai" in api_base and "/v1" in api_base:
        print("⚠️ Detected '/v1' in Tamus AI URL. Fixing to '/api'...")
        api_base = api_base.replace("/v1", "/api")
        
    print(f"\nAPI Base URL (in-script corrected): {api_base}")

    # Construct the full chat completions URL
    # If base is ".../api/" or ".../v1/", we want ".../api/chat/completions"
    if api_base.endswith("/"):
        url = f"{api_base}chat/completions"
    else:
        url = f"{api_base}/chat/completions"
        
    print(f"Target URL: {url}")

    # 2. Test Connection with multiple models
    print("\nAttempting to connect to the API with various cheap models...")
    
    # List of models to test (Using the 'protected.' prefix as found earlier)
    candidates = [
        "protected.gemini-2.0-flash-lite",
        "protected.gemini-2.5-flash-lite",
        "protected.o3-mini",
        "protected.gpt-4o-mini",
        "protected.llama-3.2",
        "protected.claude-3.5-haiku"
    ]
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    working_models = []
    
    for model in candidates:
        print(f"\nTesting model: {model} ...", end=" ")
        
        payload = {
            "model": model,
            "stream": False,
            "messages": [
                {"role": "user", "content": "Hi"}
            ]
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=10)
            
            if response.status_code == 200:
                print("✅ Success!")
                # print(f"Response: {response.json()}") # Uncomment for verbose
                working_models.append(model)
            else:
                print(f"❌ Failed (Status: {response.status_code})")
                print(f"  Reason: {response.text}")
                
        except Exception as e:
            print("❌ Exception.")
            print(f"  Error: {e}")

    print("\n---------------------------------------------------")
    
    # 3. Model Listing (Optional Check)
    # The user's curl used /api/models
    models_url = url.replace("/chat/completions", "/models")
    print(f"Attempting to list models via GET {models_url} ...")
    try:
        resp = requests.get(models_url, headers={"Authorization": f"Bearer {api_key}"}, timeout=5)
        if resp.status_code == 200:
            print("Server returned models:")
            data = resp.json()
            # Handle standard OpenAI format {data: [...]} or just list depending on implementation
            models = data.get('data', data) 
            for m in models:
                mid = m.get('id', m) if isinstance(m, dict) else m
                print(f" - {mid}")
        else:
             print(f"List models failed: {resp.status_code} - {resp.text}")
    except Exception as e:
        print(f"List models error: {e}")
        
    print("\n---------------------------------------------------")

    if working_models:
        print(f"✅ Working models: {', '.join(working_models)}")
        print(f"RECOMMENDATION: Use '{working_models[0]}' as your judge model.")
    else:
        print("❌ No models worked. Please check your API provider for exact model names.")

if __name__ == "__main__":
    verify_api()
