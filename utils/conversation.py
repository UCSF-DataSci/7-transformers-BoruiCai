# utils/conversation.py

import requests
import argparse
import os

def get_response(msg, hist=None, model="HuggingFaceH4/zephyr-7b-beta", key=None, keep=3):
    if hist is None:
        hist = []

    # build prompt from recent history
    lines = ""
    for q, a in hist[-keep:]:
        lines += f"User: {q}\nBot: {a}\n"
    lines += f"User: {msg}\nBot:"

    url = f"https://api-inference.huggingface.co/models/{model}"
    head = {"Authorization": f"Bearer {key}"} if key else {}
    data = {"inputs": lines}

    try:
        res = requests.post(url, headers=head, json=data)
        res.raise_for_status()
        out = res.json()
        if isinstance(out, list) and "generated_text" in out[0]:
            return out[0]["generated_text"].replace(lines, "").strip()
        else:
            return str(out)
    except Exception as e:
        return f"Error: {e}"

def run_chat(model, key):
    print("Chat w/ memory — type 'exit' to quit")
    hist = []
    while True:
        user = input("You: ")
        if user.lower() == "exit":
            print("Bye!")
            break
        bot = get_response(user, hist, model, key)
        hist.append((user, bot))
        print("Bot:", bot)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="HuggingFaceH4/zephyr-7b-beta")
    parser.add_argument("--api_key", default=os.getenv("HF_API_KEY"))
    args, _ = parser.parse_known_args()
    run_chat(args.model, args.api_key)

if __name__ == "__main__":
    main()