import json
import os
import boto3
from rapidfuzz import fuzz
from dotenv import load_dotenv

load_dotenv()

AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
LLM_MODEL = os.environ.get("BEDROCK_LLM_MODEL", "amazon.nova-micro-v1:0")

bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)


class SanctionsEngine:
    def __init__(self, watchlist_path="../data/finance/watchlist.json", threshold=70):
        with open(watchlist_path) as f:
            self.watchlist = json.load(f)
        self.threshold = threshold

    def fuzzy_screen(self, name):
        matches = []
        sender = name.lower().strip()

        for entry in self.watchlist:
            target = entry["name"].lower().strip()

            token_sort = fuzz.token_sort_ratio(sender, target)
            token_set = fuzz.token_set_ratio(sender, target)
            partial = fuzz.partial_ratio(sender, target)

            best_score = max(token_sort, token_set, partial)

            if best_score >= self.threshold:
                matches.append({
                    "watchlist_id": entry["id"],
                    "watchlist_name": entry["name"],
                    "watchlist_country": entry["country"],
                    "program": entry["program"],
                    "token_sort_score": token_sort,
                    "token_set_score": token_set,
                    "partial_score": partial,
                    "best_score": best_score
                })

        return sorted(matches, key=lambda x: x["best_score"], reverse=True)

    def llm_review(self, name, match):
        prompt = f"""You are a sanctions compliance analyst. Review this potential sanctions match and determine if it is a TRUE MATCH or FALSE POSITIVE.

Name to screen: {name}

Watchlist Entry:
- Name: {match["watchlist_name"]}
- Country: {match["watchlist_country"]}
- Program: {match["program"]}
- Fuzzy Match Score: {match["best_score"]}%

Consider:
1. Name similarity - are these likely the same person? Account for transliteration, spelling variations, hyphens, missing middle names.
2. Overall risk assessment.

Respond in this exact JSON format only, no other text:
{{"verdict": "TRUE_MATCH" or "FALSE_POSITIVE", "confidence": <1-100>, "risk_level": "HIGH" or "MEDIUM" or "LOW", "reasoning": "<brief explanation>"}}"""

        body = json.dumps({
            "messages": [
                {
                    "role": "user",
                    "content": [{"text": prompt}]
                }
            ],
            "inferenceConfig": {
                "max_new_tokens": 300
            }
        })

        resp = bedrock.invoke_model(
            modelId=LLM_MODEL,
            contentType="application/json",
            accept="application/json",
            body=body
        )

        output = json.loads(resp["body"].read().decode("utf-8"))
        result_text = output.get("output", {}).get("message", {}).get("content", [{}])[0].get("text", "{}")

        try:
            return json.loads(result_text)
        except json.JSONDecodeError:
            return {"verdict": "REVIEW_NEEDED", "confidence": 0, "risk_level": "HIGH", "reasoning": "LLM response could not be parsed"}

    def screen(self, name):
        matches = self.fuzzy_screen(name)

        if not matches:
            return {
                "name": name,
                "status": "CLEAR",
                "message": f"No sanctions matches found for '{name}'."
            }

        top_match = matches[0]
        review = self.llm_review(name, top_match)

        return {
            "name": name,
            "status": "FLAGGED" if review["verdict"] == "TRUE_MATCH" else "CLEAR",
            "matched_to": top_match["watchlist_name"],
            "program": top_match["program"],
            "fuzzy_score": top_match["best_score"],
            "verdict": review["verdict"],
            "confidence": review["confidence"],
            "risk_level": review["risk_level"],
            "reasoning": review["reasoning"]
        }