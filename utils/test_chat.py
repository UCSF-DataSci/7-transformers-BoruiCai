# utils/test_chat.py
import os
from one_off_chat import get_response as get_one_off_response

def test_chat(questions, model_name="HuggingFaceH4/zephyr-7b-beta", api_key=None):

    results = {}
    
    for question in questions:
        print(f"Testing question: {question}")
        response = get_one_off_response(question, model_name, api_key)
        results[question] = response
        
    return results

test_questions = [
    "What are the symptoms of gout?",
    "How is gout diagnosed?",
    "What treatments are available for gout?",
    "What lifestyle changes can help manage gout?",
    "What foods should be avoided with gout?"
]

def save_results(results, output_file="results/part_2/example.txt"):

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# LLM Chat Tool Test Results\n\n")
        f.write("## Usage Examples\n\n")
        f.write("```bash\n")
        f.write("# Run the one-off chat\n")
        f.write("python utils/one_off_chat.py\n\n")
        f.write("# Run the contextual chat\n")
        f.write("python utils/conversation.py\n")
        f.write("```\n\n")
        f.write("## Test Results\n\n")
        f.write("```csv\n")
        f.write("question,response\n")
        for question, response in results.items():
            q = question.replace(',', '').replace('\n', ' ')
            r = response.replace(',', '').replace('\n', ' ')
            f.write(f"{q},{r}\n")
        f.write("```\n")

# Run the test and save results
if __name__ == "__main__":
    model_name = "HuggingFaceH4/zephyr-7b-beta"
    api_key = os.getenv("HF_API_KEY")

    if not api_key:
        print("Please set HF_API_KEY environment variable.")
    else:
        results = test_chat(test_questions, model_name, api_key)
        save_results(results)
        print("Test results saved to results/part_2/example.txt")