import json
from pathlib import Path
from typing import List, Tuple

import torch
from torch.nn.functional import normalize
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import random

# ---------------------------------------------------------------------------
# 1. Global configuration
# ---------------------------------------------------------------------------
K = 130  # target is 128; a few extra are kept in case some are later excluded
SIM_THRESHOLD = 0.9
RANDOM_SEED = 42  # fixed seed so the random-retrieval result is reproducible
MODEL_NAME = "microsoft/unixcoder-base"
MAX_LENGTH = 512
BATCH_SIZE = 32
import config
TRAIN_FILE = config.CODE_DOC_PAIRS
TEST_FILE = config.CODE_DOC_PAIRS
OUTPUT_FILE = config.OUTPUT_DATA_DIR / "fewshot_index.json"

# ---------------------------------------------------------------------------
# 2. Model and preprocessing
# ---------------------------------------------------------------------------


def setup_model(model_name: str) -> Tuple[AutoTokenizer, AutoModel, str]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"INFO: using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return tokenizer, model, device


def extract_code_field(item: dict) -> str:
    # Read the code field directly; make sure a non-empty string is returned.
    value = item.get("code_formatted")  # uniformly formatted, comment-stripped code
    if isinstance(value, str) and value.strip():
        return value
    return ""


def normalize_code_snippet(code: str) -> str:
    if not code:
        return ""
    lines = [line.rstrip() for line in code.strip().splitlines()]
    return "\n".join(lines)


def preprocess_dataset(raw_data: List[dict], label: str) -> Tuple[List[dict], List[str], int]:
    """Drop entries with empty code; return (filtered data, normalized codes, skipped count)."""
    filtered_data = []
    processed_codes = []
    skipped = 0
    for item in raw_data:
        code = extract_code_field(item)
        if not code.strip():
            skipped += 1
            print(f"WARN: {label}: id={item.get('id')} has a missing/empty code field; skipped.")
            continue
        filtered_data.append(item)
        processed_codes.append(normalize_code_snippet(code))
    if skipped:
        print(f"WARN: {label}: {skipped} entries lack a code field; skipped.")
    return filtered_data, processed_codes, skipped


def encode_code_snippets(
    tokenizer: AutoTokenizer,
    model: AutoModel,
    codes: List[str],
    *,
    batch_size: int,
    device: str,
) -> torch.Tensor:
    if not codes:
        hidden_size = model.config.hidden_size
        return torch.empty((0, hidden_size))

    embeddings = []
    for start in tqdm(range(0, len(codes), batch_size), desc="Encoding codes"):
        batch_codes = codes[start : start + batch_size]
        inputs = tokenizer(
            batch_codes,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
        embeddings.append(cls_embeddings.cpu())
    return torch.cat(embeddings, dim=0)


# ---------------------------------------------------------------------------
# 3. Few-shot retrieval construction
# ---------------------------------------------------------------------------


def build_fewshot_output(
    test_embeddings: torch.Tensor,
    train_embeddings: torch.Tensor,
    test_data: List[dict],
    train_data: List[dict],
) -> List[dict]:
    """
    For each test entry, build both random-retrieval and semantic-retrieval
    few-shot information.

    Output format (one record per entry):
        id: id of the target entry
        fewshot_random_id:        ids of K randomly drawn non-same-project entries
        fewshot_random_cos_sim:   corresponding cosine similarities (4 decimals)
        fewshot_semantic_id:      top-K entries with similarity < threshold (descending)
        fewshot_semantic_cos_sim: corresponding cosine similarities (4 decimals)
        fewshot_semantic_unlimited_id:      top-K entries with similarity >= threshold (descending)
        fewshot_semantic_unlimited_cos_sim: corresponding cosine similarities (4 decimals)
    """
    if test_embeddings.size(0) == 0 or train_embeddings.size(0) == 0:
        print("WARN: no usable embeddings; returning an empty result.")
        return []

    train_norm = normalize(train_embeddings, p=2, dim=1)
    test_norm = normalize(test_embeddings, p=2, dim=1)

    train_ids = [item.get("id") for item in train_data]
    train_projects = [item.get("project_name") for item in train_data]

    fewshot_results = []

    for idx, test_item in enumerate(tqdm(test_data, desc="Building fewshot")):
        test_id = test_item.get("id")
        test_project = test_item.get("project_name")
        test_vec = test_norm[idx]

        # Cosine similarity between this test entry and every train entry.
        sims = torch.matmul(train_norm, test_vec).tolist()

        # ------------------------------------------------------------------
        # Build the candidate list: (sim, train_idx), excluding same-project
        # entries.
        # ------------------------------------------------------------------
        all_candidates = []
        for j, sim in enumerate(sims):
            if train_projects[j] == test_project:
                continue
            all_candidates.append((sim, j))

        # ------------------------------------------------------------------
        # Random-retrieval strategy: draw K non-same-project entries at random.
        #   Selection does not depend on similarity, but the cosine similarity
        #   is reported in the output.
        # ------------------------------------------------------------------
        random.shuffle(all_candidates)
        random_selected = all_candidates[:K]

        fewshot_random_id = [train_ids[j] for _, j in random_selected]
        fewshot_random_cos_sim = [round(sim, 4) for sim, _ in random_selected]

        # ------------------------------------------------------------------
        # Semantic-retrieval strategy: sort by similarity descending, then split
        # into "below threshold" and "at/above threshold" groups.
        # ------------------------------------------------------------------
        all_candidates.sort(key=lambda x: x[0], reverse=True)

        below_threshold = [(sim, j) for sim, j in all_candidates if sim < SIM_THRESHOLD]
        above_threshold = [(sim, j) for sim, j in all_candidates if sim >= SIM_THRESHOLD]

        # Below threshold: take the top K.
        below_selected = below_threshold[:K]
        fewshot_semantic_id = [train_ids[j] for _, j in below_selected]
        fewshot_semantic_cos_sim = [round(sim, 4) for sim, _ in below_selected]

        # At/above threshold: take the top K (usually fewer than K).
        above_selected = above_threshold[:K]
        fewshot_semantic_unlimited_id = [train_ids[j] for _, j in above_selected]
        fewshot_semantic_unlimited_cos_sim = [round(sim, 4) for sim, _ in above_selected]

        fewshot_results.append({
            "id": test_id,
            "fewshot_random_id": fewshot_random_id,
            "fewshot_random_cos_sim": fewshot_random_cos_sim,
            "fewshot_semantic_id": fewshot_semantic_id,
            "fewshot_semantic_cos_sim": fewshot_semantic_cos_sim,
            "fewshot_semantic_unlimited_id": fewshot_semantic_unlimited_id,
            "fewshot_semantic_unlimited_cos_sim": fewshot_semantic_unlimited_cos_sim,
        })

    return fewshot_results


# ---------------------------------------------------------------------------
# 4. Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    print("--- UniXcoder few-shot builder starting ---")
    tokenizer, model, device = setup_model(MODEL_NAME)

    train_path = Path(TRAIN_FILE)
    test_path = Path(TEST_FILE)
    if not train_path.exists() or not test_path.exists():
        missing = [str(p) for p in (train_path, test_path) if not p.exists()]
        print(f"ERROR: input file(s) not found: {', '.join(missing)}")
        return

    with train_path.open("r", encoding="utf-8") as f:
        train_data = json.load(f)
    with test_path.open("r", encoding="utf-8") as f:
        test_data = json.load(f)

    # Verify that every entry has a non-empty project_name.
    for label, data in [("train set", train_data), ("test set", test_data)]:
        for item in data:
            proj = item.get("project_name")
            if not proj or not str(proj).strip():
                raise ValueError(
                    f"{label}: id={item.get('id')} has an empty project_name; "
                    f"same-project exclusion cannot be applied."
                )

    print(f"INFO: train entries {len(train_data)}, test entries {len(test_data)}")

    train_data, train_codes, _ = preprocess_dataset(train_data, "train set")
    test_data, test_codes, _ = preprocess_dataset(test_data, "test set")

    print("INFO: encoding train-set code...")
    train_emb = encode_code_snippets(
        tokenizer, model, train_codes,
        batch_size=BATCH_SIZE, device=device,
    )

    print("INFO: encoding test-set code...")
    test_emb = encode_code_snippets(
        tokenizer, model, test_codes,
        batch_size=BATCH_SIZE, device=device,
    )

    random.seed(RANDOM_SEED)  # fixed seed so random retrieval is reproducible

    print(f"INFO: building few-shot information (K={K})...")
    fewshot_results = build_fewshot_output(
        test_emb, train_emb, test_data, train_data,
    )

    output_path = Path(OUTPUT_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(fewshot_results, f, ensure_ascii=False, indent=4)
    print(f"INFO: results saved to {output_path.resolve()}")
    print(f"INFO: processed {len(fewshot_results)} test entries")
    print("--- done ---")


if __name__ == "__main__":
    # dependencies: pip install torch transformers tqdm
    main()
