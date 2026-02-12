
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

    # 2. Raw CURL test (Bypass all libraries)
    print("\nAttempting Raw JSON POST (Direct to Google API)...")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    headers = {'Content-Type': 'application/json'}
    payload = {
        "contents": [{"parts": [{"text": "Hello"}]}]
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        if response.status_code == 200:
            print("✅ Raw API Success!")
            print("The issue is likely in the VLMEvalKit configuration or 'google-genai' library.")
        else:
            print(f"❌ Raw API Failure (Status: {response.status_code})")
            print(f"Response: {response.text}")
            if "API_KEY_INVALID" in response.text:
                print("\nGoogle STILL says this key is invalid.")
                print("Things to check:")
                print("1. Did you copy the key correctly from AI Studio?")
                print("2. Is the key enabled for 'Gemini API' (Standard tier)?")
                print("3. Are there any spaces in your copy-paste?")
    except Exception as e:
        print(f"❌ Connection Error: {e}")

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
