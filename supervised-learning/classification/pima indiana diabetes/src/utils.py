import pandas as pd
from config import EVAL_RESULT_DIR

def save_model_comparison(model_results, filename="model_comparison.csv"):
    
    comparison = pd.DataFrame(model_results).drop(columns=["y_proba"], errors="ignore")

    EVAL_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = EVAL_RESULT_DIR / filename
    comparison.to_csv(output_path, index=False)
    return output_path