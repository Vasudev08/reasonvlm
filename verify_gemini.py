
import os
import sys

# Add VLMEvalKit to path
script_path = os.path.abspath(__file__)
root_dir = os.path.dirname(script_path)
vlm_kit_path = os.path.join(root_dir, "VLMEvalKit")
sys.path.insert(0, vlm_kit_path)

def verify_gemini():
    print("--- 🕵️ Verifying Google Gemini API Configuration ---")
    
    api_key = os.environ.get("GOOGLE_API_KEY")
    print(f"Checking Environment Variables:")
    print(f"  GOOGLE_API_KEY: {'✅ Set' if api_key else '❌ Not Set'}")
    
    if not api_key:
        print("\n❌ Error: GOOGLE_API_KEY is missing.")
        print("Please run: os.environ['GOOGLE_API_KEY'] = 'your_key'")
        return

    try:
        # VLMEvalKit uses google-genai or vertexai depending on config.
        # But for judging, it usually uses its own Gemini wrapper.
        from vlmeval.api import Gemini
        
        # Test with Gemini-1.5-Flash (very cheap/fast)
        print("\nTesting connection with GeminiFlash1-5...")
        model = Gemini(model="gemini-1.5-flash", retry=3, verbose=True)
        
        response = model.generate("Hello! Just a test. Reply with 'Gemini is online'.")
        
        print("\n---------------------------------------------------")
        if response:
            print(f"✅ Success! API responded: {response}")
            print("Your native Google Gemini API is working correctly.")
        else:
            print("⚠️ Empty response. Check your API key permissions/quota.")
            
    except Exception as e:
        print(f"\n❌ Gemini API Call Failed!")
        print(f"Error: {e}")
        print("\nMake sure you have installed 'google-genai' and your key is valid.")

if __name__ == "__main__":
    verify_gemini()
