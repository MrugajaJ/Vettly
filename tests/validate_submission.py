import sys
import re
import polars as pl

def validate_submission(csv_path: str):
    """Validates the structure and content of the submission CSV file."""
    print(f"Validating submission CSV: {csv_path}")
    
    # Load the CSV using Polars
    try:
        df = pl.read_csv(csv_path)
    except Exception as e:
        print(f"Error: Failed to read CSV: {e}")
        sys.exit(1)
        
    # 1. Assert row count <= 100
    row_count = df.height
    print(f"Row count: {row_count}")
    assert row_count <= 100, f"Row count {row_count} exceeds maximum of 100"
    
    if row_count == 0:
        print("Warning: Submission is empty.")
        return
        
    # 2. Assert rank column is sequential 1..N with no gaps
    ranks = df["rank"].to_list()
    expected_ranks = list(range(1, row_count + 1))
    assert ranks == expected_ranks, f"Ranks are not sequential 1..{row_count}. Found: {ranks}"
    
    # 3. Assert all candidate_id values match regex CAND_[0-9]{7}
    candidate_ids = df["candidate_id"].to_list()
    id_pattern = re.compile(r"^CAND_[0-9]{7}$")
    for cid in candidate_ids:
        assert id_pattern.match(str(cid)), f"Invalid candidate_id format: '{cid}' (expected 'CAND_1234567')"
        
    # 4. Assert all final_score values are in [0.0, 1.0]
    final_scores = df["final_score"].to_list()
    for i, score in enumerate(final_scores):
        score_val = float(score)
        assert 0.0 <= score_val <= 1.0, f"Score {score_val} at rank {i+1} is out of bounds [0.0, 1.0]"
        
    print("All validation checks passed.")

if __name__ == "__main__":
    path = "submission.csv"
    if len(sys.argv) > 1:
        path = sys.argv[1]
    validate_submission(path)
