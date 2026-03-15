import json
import pandas as pd
from sanctions_engine import SanctionsEngine


def run_screening():
    engine = SanctionsEngine()

    with open("../data/finance/transactions.json") as f:
        transactions = json.load(f)

    print("=" * 70)
    print("SANCTIONS SCREENING SYSTEM - Two-Layer Pipeline")
    print("=" * 70)
    print(f"Watchlist entries: {len(engine.watchlist)}")
    print(f"Transactions to screen: {len(transactions)}")
    print(f"Fuzzy threshold: {engine.threshold}%")
    print("=" * 70)

    all_results = []

    for txn in transactions:
        print(f"\n--- Screening TXN {txn['txn_id']}: {txn['sender']} ---")

        result = engine.screen(txn["sender"])

        if result["status"] == "CLEAR" and "matched_to" not in result:
            print(f"  Layer 1: CLEAR - No fuzzy matches found")
        else:
            print(f"  Layer 1: FLAGGED - Matched '{result.get('matched_to', 'N/A')}' (score: {result.get('fuzzy_score', 0)}%)")
            print(f"  Layer 2: {result.get('verdict', 'N/A')} (confidence: {result.get('confidence', 0)}%, risk: {result.get('risk_level', 'N/A')})")
            print(f"  Reason: {result.get('reasoning', 'N/A')}")

        all_results.append({
            "txn_id": txn["txn_id"],
            "sender": txn["sender"],
            "amount": txn["amount"],
            "currency": txn["currency"],
            "country": txn["country"],
            "status": result["status"],
            "matched_to": result.get("matched_to", None),
            "fuzzy_score": result.get("fuzzy_score", 0),
            "verdict": result.get("verdict", None),
            "confidence": result.get("confidence", None),
            "risk_level": result.get("risk_level", "NONE"),
            "reasoning": result.get("reasoning", "No match")
        })

    # SUMMARY
    df = pd.DataFrame(all_results)
    flagged = df[df["verdict"] == "TRUE_MATCH"]
    cleared = df[df["status"] == "CLEAR"]

    print("\n" + "=" * 70)
    print("SCREENING SUMMARY")
    print("=" * 70)
    print(f"Total transactions:   {len(transactions)}")
    print(f"Cleared:              {len(cleared)}")
    print(f"Flagged (true match): {len(flagged)}")

    df.to_csv("screening_results.csv", index=False)
    print("\nFull results saved to screening_results.csv")

    if not flagged.empty:
        print("\nFLAGGED TRANSACTIONS:")
        for _, row in flagged.iterrows():
            print(f"  {row['txn_id']}: {row['sender']} -> '{row['matched_to']}' | {row['amount']} {row['currency']} | Risk: {row['risk_level']}")


if __name__ == "__main__":
    run_screening()