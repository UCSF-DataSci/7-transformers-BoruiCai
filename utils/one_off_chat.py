# utils/one_off_chat.py
import requests
import argparse
import os

def get_response(prompt, model_name="HuggingFaceH4/zephyr-7b-beta", api_key=None):
    url = f"https://api-inference.huggingface.co/models/{model_name}"
    head = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    data = {
    "inputs": prompt,
    "parameters": {
        "max_new_tokens": 50  
    }}
    try:
        res = requests.post(url, headers=head, json=data)
        res.raise_for_status()
        out = res.json()

        if isinstance(out, list) and "generated_text" in out[0]:
            return out[0]["generated_text"]
        else:
            return str(out)
    except Exception as e:
        return f"Error: {e}"

def run_chat(model_name, api_key):
    print("Simple LLM Chat — type 'exit' to quit")
    while True:
        msg = input("You: ")
        if msg.lower() == "exit":
            print("Bye!")
            break
        reply = get_response(msg, model_name=model_name, api_key=api_key)
        print("Bot:", reply)

def main():
    parser = argparse.ArgumentParser(description="Chat with an LLM")
    parser.add_argument("--model", default="HuggingFaceH4/zephyr-7b-beta")
    parser.add_argument("--api_key", default=os.getenv("HF_API_KEY"))
    args, _ = parser.parse_known_args()

    run_chat(model_name=args.model, api_key=args.api_key)

if __name__ == "__main__":
    main()