
import os
import sys

# Add VLMEvalKit to path
script_path = os.path.abspath(__file__)
root_dir = os.path.dirname(script_path)
vlm_kit_path = os.path.join(root_dir, "VLMEvalKit")
sys.path.insert(0, vlm_kit_path)

def verify_openai():
    print("--- 🤖 Verifying OpenAI API Configuration ---")
    
    api_key = os.environ.get("OPENAI_API_KEY")
    print(f"Checking Environment Variables:")
    print(f"  OPENAI_API_KEY: {'✅ Set' if api_key else '❌ Not Set'}")
    
    if not api_key:
        print("\n❌ Error: OPENAI_API_KEY is missing.")
        print("Please run: os.environ['OPENAI_API_KEY'] = 'your_key'")
        return

    try:
        from vlmeval.api import GPT4V
        
        # Test with GPT-4o-mini (very cheap judge)
        print("\nTesting connection with gpt-4o-mini...")
        # VLMEvalKit's GPT4V class handles OpenAI compatible APIs
        model = GPT4V(model="gpt-4o-mini", retry=3, verbose=True)
        
        response = model.generate("Hello! Just a test. Reply with 'OpenAI is online'.")
        
        print("\n---------------------------------------------------")
        if response and model.fail_msg not in response:
            print(f"✅ Success! API responded: {response}")
            print("Your OpenAI API is working correctly.")
        else:
            print(f"❌ Failure. API responded: {response}")
            print("Please check your OpenAI API key and billing/quota.")
            
    except Exception as e:
        print(f"\n❌ OpenAI API Call Failed!")
        print(f"Error: {e}")

if __name__ == "__main__":
    verify_openai()
