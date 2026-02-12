
import os
import sys
import requests

# Add VLMEvalKit to path
script_path = os.path.abspath(__file__)
root_dir = os.path.dirname(script_path)
vlm_kit_path = os.path.join(root_dir, "VLMEvalKit")
sys.path.insert(0, vlm_kit_path)

def diagnose_gemini():
    print("--- � Gemini API Diagnostic Tool ---")
    
    api_key = os.environ.get("GOOGLE_API_KEY")
    
    if not api_key:
        print("❌ Error: GOOGLE_API_KEY is NOT set in this environment.")
        return

    # 1. Inspect the key structure (safely)
    key_len = len(api_key)
    masked_key = f"{api_key[:4]}...{api_key[-4:]}" if key_len > 8 else "****"
    print(f"Key Found: {masked_key}")
    print(f"Key Length: {key_len} characters")
    
    if api_key != api_key.strip():
        print("⚠️  WARNING: Your API key has leading or trailing WHITESPACE (spaces or newlines).")
        print("   This will cause it to fail. Please .strip() your key.")
        api_key = api_key.strip()
        os.environ["GOOGLE_API_KEY"] = api_key

    # 2. Raw Model Listing (Check what is actually available)
    print("\nAttempting to List Models (Check permissions)...")
    list_url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    try:
        list_resp = requests.get(list_url, timeout=10)
        if list_resp.status_code == 200:
            print("✅ Successfully listed models!")
            models_data = list_resp.json().get('models', [])
            print(f"Your key has access to {len(models_data)} models.")
            # Print first 5 models
            for m in models_data[:5]:
                print(f" - {m.get('name')}")
        else:
            print(f"❌ Failed to list models (Status: {list_resp.status_code})")
            print(f"Response: {list_resp.text}")
    except Exception as e:
        print(f"❌ Error listing models: {e}")

    # 3. Raw CURL test (Try v1 and v1beta)
    for version in ["v1", "v1beta"]:
        print(f"\nAttempting Raw JSON POST (Direct to Google API {version})...")
        # Testing Gemini 1.5 and 2.0 (Flash and Lite)
        candidate_models = [
            "gemini-2.0-flash-lite",
            "gemini-2.0-flash",
            "gemini-1.5-flash",
            "gemini-1.5-flash-latest"
        ]
        
        for model_id in candidate_models:
            url = f"https://generativelanguage.googleapis.com/{version}/models/{model_id}:generateContent?key={api_key}"
            headers = {'Content-Type': 'application/json'}
            payload = {
                "contents": [{"parts": [{"text": "Hello"}]}]
            }
            
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=10)
                if response.status_code == 200:
                    print(f"✅ Raw API Success ({version}, {model_id})!")
                    break
                else:
                    # Don't spam 404s, just report other errors
                    if response.status_code != 404:
                         print(f"❌ Raw API Failure ({version}, {model_id}) (Status: {response.status_code})")
                         print(f"Response: {response.text}")
                    else:
                         print(f"   - {model_id} ({version}): 404 Not Found")
            except Exception as e:
                print(f"❌ Connection Error ({version}, {model_id}): {e}")
        else:
            continue
        break

    # 3. VLMEvalKit Wrapper Test
    print("\nTesting via VLMEvalKit Wrapper...")
    try:
        from vlmeval.api import Gemini
        model = Gemini(model="gemini-1.5-flash", retry=1, verbose=True)
        resp = model.generate("Hi")
        if resp and model.fail_msg not in resp:
            print(f"✅ VLMEvalKit Success: {resp}")
        else:
            print(f"❌ VLMEvalKit Failed: {resp}")
    except Exception as e:
        print(f"❌ VLMEvalKit Error: {e}")

if __name__ == "__main__":
    diagnose_gemini()
