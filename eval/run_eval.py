import json
import requests
import boto3
import os
from dotenv import load_dotenv

load_dotenv()

API_URL = os.environ.get("API_URL", "http://localhost:8000")
AWS_REGION = os.environ.get("AWS_REGION")
JUDGE_MODEL = os.environ.get("BEDROCK_LLM_MODEL")

bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)


def get_rag_answer(question):
    resp = requests.post(
        f"{API_URL}/query",
        json={"query": question},
        timeout=60
    )
    data = resp.json()
    return data.get("answer", ""), data.get("retrieved_chunks", [])


def judge_answer(question, expected, actual, chunks):
    prompt = f"""You are an evaluation judge. Score the following RAG system response.

Question: {question}
Expected Answer: {expected}
Actual Answer: {actual}
Retrieved Context: {json.dumps(chunks[:3])}

Score each dimension from 1 to 5:
1. Correctness - Does the actual answer match the expected answer?
2. Groundedness - Is the answer based on the retrieved context, not hallucinated?
3. Relevance - Are the retrieved chunks relevant to the question?

Respond in this exact JSON format only, no other text:
{{"correctness": <1-5>, "groundedness": <1-5>, "relevance": <1-5>, "explanation": "<brief reason>"}}"""

    body = json.dumps({
        "messages": [
            {
                "role": "user",
                "content": [{"text": prompt}]
            }
        ],
        "inferenceConfig": {
            "max_new_tokens": 500
        }
    })

    resp = bedrock.invoke_model(
        modelId=JUDGE_MODEL,
        contentType="application/json",
        accept="application/json",
        body=body
    )

    output = json.loads(resp["body"].read().decode("utf-8"))
    result_text = output.get("output", {}).get("message", {}).get("content", [{}])[0].get("text", "{}")
    return json.loads(result_text)


def run_eval():
    with open("eval/eval_questions.json") as f:
        questions = json.load(f)

    results = []
    total_correct = 0
    total_grounded = 0
    total_relevant = 0

    for i, q in enumerate(questions):
        print(f"\nEvaluating {i+1}/{len(questions)}: {q['question']}")

        actual_answer, chunks = get_rag_answer(q["question"])
        scores = judge_answer(
            q["question"],
            q["expected_answer"],
            actual_answer,
            chunks
        )

        total_correct += scores["correctness"]
        total_grounded += scores["groundedness"]
        total_relevant += scores["relevance"]

        results.append({
            "question": q["question"],
            "expected": q["expected_answer"],
            "actual": actual_answer,
            "scores": scores
        })

        print(f"  Correctness: {scores['correctness']}/5")
        print(f"  Groundedness: {scores['groundedness']}/5")
        print(f"  Relevance: {scores['relevance']}/5")
        print(f"  Reason: {scores['explanation']}")

    n = len(questions)
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Questions evaluated: {n}")
    print(f"Avg Correctness:  {total_correct/n:.1f}/5")
    print(f"Avg Groundedness: {total_grounded/n:.1f}/5")
    print(f"Avg Relevance:    {total_relevant/n:.1f}/5")

    with open("eval/eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nFull results saved to eval/eval_results.json")


if __name__ == "__main__":
    run_eval()