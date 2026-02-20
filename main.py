from evaluation import evaluate

if __name__ == "__main__":
    s_r = "He went to the market"      #English reference sentence
    s_cm = "Woh market gaya kiya"      #code-mixed sentence to evaluate

    score = evaluate(s_r, s_cm)
    print(f"GAME Score: {score:.4f}")
