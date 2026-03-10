"""
CRF Extraction Runner — extractor-first with checkpointing and scoring.

Usage:
    # Run Italian dev set
    python run.py --model openrouter/qwen/qwen3-235b-a22b-thinking-2507 --lang it

    # Run English dev set
    python run.py --model openrouter/qwen/qwen3-235b-a22b-thinking-2507 --lang en

    # Run both languages
    python run.py --model openrouter/qwen/qwen3-235b-a22b-thinking-2507 --lang both

    # Resume from checkpoint (auto-detects where it left off)
    python run.py --model openrouter/qwen/qwen3-235b-a22b-thinking-2507 --lang it --resume

    # Run a subset of documents
    python run.py --model openrouter/qwen/qwen3-235b-a22b-thinking-2507 --lang it --n 5

    # Score an existing submission
    python run.py --score-only submissions/submission_it.jsonl --lang it

    # Disable chain-of-thought (use dspy.Predict instead of ChainOfThought)
    python run.py --model openrouter/qwen/qwen3-235b-a22b-thinking-2507 --lang it --no-cot
"""

import argparse
import json
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path

os.environ.setdefault("PYTHONUNBUFFERED", "1")
sys.stdout.reconfigure(line_buffering=True)

import dspy
from datasets import load_dataset

from pipeline import GROUP_SIGNATURES, GROUP_OUTPUT_FIELD, configure_lm
from schema import CRFOutput, crf_to_annotations, dataset_row_to_crf, ANNOTATION_KEY_TO_FIELD

# ── Constants ─────────────────────────────────────────────────────────────────

EXTRACTOR_ORDER = list(GROUP_SIGNATURES.keys())  # 14 extractors

SCORING_REPO = Path(__file__).parent / "scoring_repo"
DEV_GT_PATH = SCORING_REPO / "development_data" / "dev_gt.jsonl"
SUBMISSIONS_DIR = Path(__file__).parent / "submissions"
CHECKPOINTS_DIR = Path(__file__).parent / "checkpoints"


# ── Data loading ──────────────────────────────────────────────────────────────

def get_canonical_item_order(test: bool = False) -> list[str]:
    """Get the exact item ordering from the ground truth file or test dataset."""
    if test:
        # For test mode, get item order from the test dataset annotations
        ds = load_dataset("NLP-FBK/dyspnea-crf-test", split="en")
        return [a["item"] for a in ds[0]["annotations"]]
    with open(DEV_GT_PATH, encoding="utf-8") as f:
        first_record = json.loads(f.readline())
    return [a["item"] for a in first_record["annotations"]]


def load_documents(lang: str, n: int = 0, test: bool = False) -> list[dict]:
    """Load clinical notes from HuggingFace.

    Returns list of {"document_id": str, "clinical_note": str, "annotations": list}.
    """
    dataset_name = "NLP-FBK/dyspnea-crf-test" if test else "NLP-FBK/dyspnea-crf-development"
    ds = load_dataset(dataset_name, split=lang)
    if n > 0:
        ds = ds.select(range(min(n, len(ds))))

    docs = []
    for row in ds:
        docs.append({
            "document_id": row["document_id"],  # e.g. "1014081_it"
            "clinical_note": row["clinical_note"],
            "annotations": row.get("annotations", []),
        })
    return docs


# ── Checkpoint management ─────────────────────────────────────────────────────

def checkpoint_path(model: str, lang: str, test: bool = False) -> Path:
    """Generate checkpoint file path based on model and language."""
    safe_model = model.replace("/", "_").replace(":", "_")
    prefix = "test_checkpoint" if test else "checkpoint"
    return CHECKPOINTS_DIR / f"{prefix}_{safe_model}_{lang}.json"


def load_checkpoint(path: Path) -> dict:
    """Load checkpoint if it exists."""
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def save_checkpoint(path: Path, data: dict):
    """Save checkpoint to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write to temp file then rename for atomicity
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def init_checkpoint(model: str, lang: str, doc_ids: list[str]) -> dict:
    """Create a new checkpoint structure."""
    return {
        "meta": {
            "model": model,
            "language": lang,
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "doc_count": len(doc_ids),
        },
        "doc_ids": doc_ids,
        "completed_extractors": [],
        "results": {},  # {doc_id: {extractor_name: {field: value, ...}}}
    }


# ── Extractor-first runner ────────────────────────────────────────────────────

def _extract_one_doc(predictor, output_field, doc_id, note):
    """Extract one document with one predictor. Thread-safe."""
    try:
        prediction = predictor(clinical_note=note)
        sub_model = getattr(prediction, output_field)
        return doc_id, sub_model.model_dump(), None
    except Exception as e:
        return doc_id, {}, str(e)


def _run_single_extractor(
    ext_name: str,
    ext_idx: int,
    docs: list[dict],
    cp: dict,
    cp_path: Path,
    cp_lock,
    optimized_program,
    predictor_cls,
    few_shot_demos: dict,
    workers: int,
):
    """Run a single extractor across all documents. Thread-safe with cp_lock."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    output_field = GROUP_OUTPUT_FIELD[ext_name]

    # Use optimized predictor if available, otherwise create fresh
    if optimized_program:
        predictor = getattr(optimized_program, f"extract_{ext_name}")
    else:
        sig_cls = GROUP_SIGNATURES[ext_name]
        predictor = predictor_cls(sig_cls)

    # Add few-shot demos if requested
    if few_shot_demos.get(ext_name):
        inner = getattr(predictor, "predict", predictor)
        inner.demos = few_shot_demos[ext_name]

    # Filter to docs that still need this extractor
    with cp_lock:
        pending_docs = [
            doc for doc in docs
            if not (doc["document_id"] in cp["results"]
                    and ext_name in cp["results"][doc["document_id"]])
        ]

    print(f"\n[{ext_idx+1}/14] {ext_name} — extracting from {len(pending_docs)} documents (workers={workers})...")
    ext_start = time.time()
    ext_errors = 0
    ext_calls = 0
    done_count = 0

    if workers <= 1:
        # Sequential execution
        for doc in pending_docs:
            doc_id, result, error = _extract_one_doc(
                predictor, output_field, doc["document_id"], doc["clinical_note"]
            )
            ext_calls += 1
            if error:
                print(f"    [{ext_name}] ERROR doc {doc_id}: {error}")
                ext_errors += 1

            with cp_lock:
                if doc_id not in cp["results"]:
                    cp["results"][doc_id] = {}
                cp["results"][doc_id][ext_name] = result
            done_count += 1

            if done_count % 10 == 0 or done_count == len(pending_docs):
                elapsed = time.time() - ext_start
                rate = done_count / elapsed if elapsed > 0 else 0
                print(f"    [{ext_name}] [{done_count}/{len(pending_docs)}] {rate:.1f} docs/sec, {elapsed:.0f}s elapsed")
            if done_count % 20 == 0:
                with cp_lock:
                    save_checkpoint(cp_path, cp)
    else:
        # Parallel execution
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _extract_one_doc, predictor, output_field,
                    doc["document_id"], doc["clinical_note"]
                ): doc["document_id"]
                for doc in pending_docs
            }

            for future in as_completed(futures):
                doc_id, result, error = future.result()
                ext_calls += 1
                if error:
                    print(f"    [{ext_name}] ERROR doc {doc_id}: {error}")
                    ext_errors += 1
                    done_count += 1
                    continue

                with cp_lock:
                    if doc_id not in cp["results"]:
                        cp["results"][doc_id] = {}
                    cp["results"][doc_id][ext_name] = result
                done_count += 1

                if done_count % 10 == 0 or done_count == len(pending_docs):
                    elapsed = time.time() - ext_start
                    rate = done_count / elapsed if elapsed > 0 else 0
                    print(f"    [{ext_name}] [{done_count}/{len(pending_docs)}] {rate:.1f} docs/sec, {elapsed:.0f}s elapsed")
                if done_count % 20 == 0:
                    with cp_lock:
                        save_checkpoint(cp_path, cp)

    # Mark extractor as complete only if no errors
    with cp_lock:
        if ext_errors == 0:
            cp["completed_extractors"].append(ext_name)
        save_checkpoint(cp_path, cp)

    ext_elapsed = time.time() - ext_start
    print(f"    [{ext_name}] Done: {ext_elapsed:.1f}s" + (f" ({ext_errors} errors)" if ext_errors else ""))
    return ext_calls, ext_errors


def run_extraction(
    model: str,
    lang: str,
    docs: list[dict],
    resume: bool = False,
    use_cot: bool = True,
    temperature: float = 0.0,
    optimized_path: str = None,
    workers: int = 1,
    few_shot: int = 0,
    test: bool = False,
    parallel_extractors: int = 1,
):
    """Run all 14 extractors across all documents, extractor-first.

    This keeps the same system prompt + few-shot demos in cache across documents.
    Saves checkpoint after each extractor completes all documents.

    Args:
        workers: Number of parallel threads per extractor for document processing.
                 1 = sequential (default), >1 = concurrent API calls.
        parallel_extractors: Number of extractors to run concurrently.
                 1 = sequential (default), >1 = multiple extractors run at once.
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    doc_ids = [d["document_id"] for d in docs]
    doc_map = {d["document_id"]: d["clinical_note"] for d in docs}

    cp_path = checkpoint_path(model, lang, test=test)

    # Load or create checkpoint
    if resume:
        cp = load_checkpoint(cp_path)
        if cp is None:
            print("No checkpoint found, starting fresh.")
            cp = init_checkpoint(model, lang, doc_ids)
        else:
            done = cp["completed_extractors"]
            print(f"Resuming from checkpoint: {len(done)}/14 extractors done")
            print(f"  Completed: {', '.join(done)}")
            # Verify doc_ids match
            if cp["doc_ids"] != doc_ids:
                print("WARNING: Document list changed since checkpoint. Starting fresh.")
                cp = init_checkpoint(model, lang, doc_ids)
    else:
        cp = init_checkpoint(model, lang, doc_ids)

    # Configure LM
    print(f"\nModel:       {model}")
    print(f"Language:    {lang}")
    print(f"Documents:   {len(docs)}")
    print(f"CoT:         {use_cot}")
    print(f"Temperature: {temperature}")
    print(f"Workers:     {workers}")
    print(f"Par. Ext.:   {parallel_extractors}")
    print(f"Optimized:   {optimized_path or 'No'}")
    print(f"Few-shot:    {few_shot if few_shot else 'No'}")
    # Qwen3 models (non-thinking variants) need thinking disabled for JSON mode
    lm_kwargs = dict(temperature=temperature)
    if "qwen3" in model.lower() and "thinking" not in model.lower():
        lm_kwargs["extra_body"] = {"enable_thinking": False}
        print("Note:        Disabled native thinking for JSON mode compatibility")
    configure_lm(model, **lm_kwargs)

    # Load few-shot demos from training data if requested
    few_shot_demos = {}  # {ext_name: [dspy.Example, ...]}
    if few_shot > 0:
        from optimize_per_extractor import load_single_extractor_examples
        for ext_name in EXTRACTOR_ORDER:
            exs = load_single_extractor_examples("train", lang, ext_name, filter_known=True)
            # Take up to `few_shot` examples
            few_shot_demos[ext_name] = exs[:few_shot]
        print(f"Loaded few-shot demos from training data")

    # Load optimized program or create fresh predictors
    optimized_program = None
    if optimized_path:
        if optimized_path.startswith("dir:"):
            # Per-extractor optimized directory
            from pipeline import compose_optimized_extractors
            opt_dir = optimized_path[4:]
            optimized_program = compose_optimized_extractors(opt_dir, use_cot=use_cot)
            print(f"Loaded per-extractor optimized program from: {opt_dir}")
        else:
            from optimize import load_optimized_program
            optimized_program = load_optimized_program(optimized_path, use_cot=use_cot)

    predictor_cls = dspy.ChainOfThought if use_cot else dspy.Predict

    # Identify pending extractors
    pending_extractors = [
        (ext_idx, ext_name) for ext_idx, ext_name in enumerate(EXTRACTOR_ORDER)
        if ext_name not in cp["completed_extractors"]
    ]
    skipped = len(EXTRACTOR_ORDER) - len(pending_extractors)
    if skipped:
        print(f"\nSkipping {skipped} already-completed extractors.")

    # Thread-safe lock for checkpoint access
    cp_lock = threading.Lock()

    total_calls = 0
    total_errors = 0
    start_time = time.time()

    if parallel_extractors <= 1:
        # Sequential extractor execution (original behavior)
        for ext_idx, ext_name in pending_extractors:
            calls, errors = _run_single_extractor(
                ext_name, ext_idx, docs, cp, cp_path, cp_lock,
                optimized_program, predictor_cls, few_shot_demos, workers,
            )
            total_calls += calls
            total_errors += errors
    else:
        # Parallel extractor execution — run N extractors concurrently
        print(f"\nRunning extractors in parallel (batch size={parallel_extractors})...")

        # Process in batches of parallel_extractors
        for batch_start in range(0, len(pending_extractors), parallel_extractors):
            batch = pending_extractors[batch_start:batch_start + parallel_extractors]
            batch_names = [name for _, name in batch]
            print(f"\n{'='*60}")
            print(f"  Parallel batch: {', '.join(batch_names)}")
            print(f"{'='*60}")

            with ThreadPoolExecutor(max_workers=parallel_extractors) as ext_executor:
                ext_futures = {
                    ext_executor.submit(
                        _run_single_extractor,
                        ext_name, ext_idx, docs, cp, cp_path, cp_lock,
                        optimized_program, predictor_cls, few_shot_demos, workers,
                    ): ext_name
                    for ext_idx, ext_name in batch
                }

                for future in as_completed(ext_futures):
                    ext_name = ext_futures[future]
                    try:
                        calls, errors = future.result()
                        total_calls += calls
                        total_errors += errors
                    except Exception as e:
                        print(f"    [{ext_name}] FATAL ERROR: {e}")
                        total_errors += 1

    total_elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  Extraction complete!")
    print(f"  Total calls: {total_calls}")
    print(f"  Total errors: {total_errors}")
    print(f"  Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"  Checkpoint: {cp_path}")
    print(f"{'='*60}")

    return cp


# ── Submission generation ─────────────────────────────────────────────────────

def checkpoint_to_submission(cp: dict, lang: str, test: bool = False) -> list[dict]:
    """Convert checkpoint results to submission JSONL format.

    Critically: items must be in the exact same order as the ground truth.
    """
    canonical_items = get_canonical_item_order(test=test)

    # Build reverse mapping: annotation_key → (group, field)
    key_to_field = ANNOTATION_KEY_TO_FIELD

    # For each document, assemble a CRFOutput from checkpoint data,
    # then convert to annotations in canonical order.
    submission = []

    for doc_id in cp["doc_ids"]:
        doc_results = cp["results"].get(doc_id, {})

        # Rebuild CRFOutput from checkpoint
        crf_kwargs = {}
        for ext_name in EXTRACTOR_ORDER:
            ext_data = doc_results.get(ext_name, {})
            crf_kwargs[ext_name] = ext_data

        try:
            crf = CRFOutput(**crf_kwargs)
        except Exception:
            crf = CRFOutput()

        # Convert to annotations format
        annotations = crf_to_annotations(crf)
        ann_map = {a["item"]: a["ground_truth"] for a in annotations}

        # Build predictions in canonical order
        predictions = []
        for item_name in canonical_items:
            pred_value = ann_map.get(item_name, "unknown")
            predictions.append({
                "item": item_name,
                "prediction": pred_value,
            })

        submission.append({
            "document_id": doc_id,  # already has _lang suffix from HF
            "predictions": predictions,
        })

    return submission


def save_submission(submission: list[dict], model: str, lang: str, test: bool = False) -> Path:
    """Save submission as JSONL file."""
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    safe_model = model.replace("/", "_").replace(":", "_")
    prefix = "test_submission" if test else "submission"
    filename = f"{prefix}_{safe_model}_{lang}.jsonl"
    path = SUBMISSIONS_DIR / filename

    with open(path, "w", encoding="utf-8") as f:
        for record in submission:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Submission saved: {path}")
    print(f"  Documents: {len(submission)}")
    print(f"  Items per doc: {len(submission[0]['predictions']) if submission else 0}")
    return path


# ── Scoring ───────────────────────────────────────────────────────────────────

def generate_ground_truth(lang: str, doc_ids: list[str]) -> Path:
    """Generate ground truth JSONL from HuggingFace in the scorer's format.

    The scorer expects document_id WITHOUT language suffix.
    """
    ds = load_dataset("NLP-FBK/dyspnea-crf-development", split=lang)
    doc_id_set = set(doc_ids)

    gt_path = SUBMISSIONS_DIR / f"dev_gt_{lang}.jsonl"
    with open(gt_path, "w", encoding="utf-8") as f:
        for row in ds:
            hf_doc_id = row["document_id"]  # e.g. "1014081_it"
            if hf_doc_id not in doc_id_set:
                continue
            # Strip language suffix for ground truth format
            base_id = hf_doc_id.rsplit("_", 1)[0]
            record = {
                "document_id": base_id,
                "annotations": row["annotations"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return gt_path


def run_scoring(submission_path: Path, lang: str, gt_path: Path = None):
    """Run the official scorer against a submission file."""
    # Import scorer
    sys.path.insert(0, str(SCORING_REPO))
    from scoring import Scorer, load_jsonl

    if gt_path is None:
        gt_path = DEV_GT_PATH

    print(f"\n{'='*60}")
    print(f"  SCORING")
    print(f"  Submission: {submission_path}")
    print(f"  Ground truth: {gt_path}")
    print(f"  Language: {lang}")
    print(f"{'='*60}")

    reference = load_jsonl(str(gt_path))
    submission = load_jsonl(str(submission_path))

    # Verify counts
    print(f"  Reference docs: {len(reference)}")
    print(f"  Submission docs: {len(submission)}")

    if len(reference) != len(submission):
        print(f"  WARNING: document count mismatch!")
        # Try to align by matching IDs
        ref_ids = {r["document_id"] for r in reference}
        sub_base_ids = {s["document_id"].rsplit("_", 1)[0] for s in submission}
        common = ref_ids & sub_base_ids
        print(f"  Common IDs: {len(common)}")
        if len(common) < len(reference):
            missing = ref_ids - sub_base_ids
            print(f"  Missing from submission: {missing}")

    scorer = Scorer(not_available_string="unknown", language=lang)
    try:
        score = scorer.calculate_score(reference, submission)
        print(f"\n  *** Macro-F1 = {score:.4f} ({score*100:.2f}%) ***\n")

        # Save score
        scores_dir = SUBMISSIONS_DIR / "scores"
        scores_dir.mkdir(parents=True, exist_ok=True)
        safe_name = submission_path.stem
        scores_path = scores_dir / f"{safe_name}_scores.json"
        with open(scores_path, "w") as f:
            json.dump({"f1_macro": float(score), "submission": str(submission_path)}, f, indent=2)
        print(f"  Score saved: {scores_path}")

        return score
    except Exception as e:
        print(f"  Scoring error: {e}")
        return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CRF extraction runner with checkpointing and scoring"
    )
    parser.add_argument("--model", default="openrouter/qwen/qwen3-235b-a22b-thinking-2507",
                        help="LiteLLM model string")
    parser.add_argument("--lang", choices=["en", "it", "both"], default="it",
                        help="Language split to process")
    parser.add_argument("--n", type=int, default=0,
                        help="Number of documents (0=all)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint")
    parser.add_argument("--no-cot", action="store_true",
                        help="Disable chain-of-thought")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="LM temperature")
    parser.add_argument("--score-only", type=str, default=None,
                        help="Score an existing submission file (skip extraction)")
    parser.add_argument("--no-score", action="store_true",
                        help="Skip scoring after extraction")
    parser.add_argument("--optimized", type=str, default=None,
                        help="Path to optimized DSPy program JSON file")
    parser.add_argument("--optimized-dir", type=str, default=None,
                        help="Path to per-extractor optimized directory (from optimize_per_extractor.py)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers per extractor (1=sequential, 5-10 recommended)")
    parser.add_argument("--few-shot", type=int, default=0,
                        help="Number of labeled few-shot demos per extractor (0=none)")
    parser.add_argument("--test", action="store_true",
                        help="Run on test dataset (NLP-FBK/dyspnea-crf-test) instead of dev")
    parser.add_argument("--parallel-extractors", type=int, default=1,
                        help="Number of extractors to run concurrently (1=sequential, 2+=parallel)")

    args = parser.parse_args()

    # Score-only mode
    if args.score_only:
        sub_path = Path(args.score_only)
        if not sub_path.exists():
            print(f"Submission file not found: {sub_path}")
            sys.exit(1)
        langs = ["en", "it"] if args.lang == "both" else [args.lang]
        for lang in langs:
            # Use the official dev_gt.jsonl for Italian, generate for English
            if lang == "it":
                run_scoring(sub_path, lang, DEV_GT_PATH)
            else:
                gt_path = generate_ground_truth(lang, doc_ids=None)
                run_scoring(sub_path, lang, gt_path)
        return

    # Determine languages to process
    langs = ["en", "it"] if args.lang == "both" else [args.lang]

    is_test = args.test
    if is_test:
        print("\n*** TEST MODE: Loading from NLP-FBK/dyspnea-crf-test ***")
        print("*** Scoring will be skipped (no ground truth) ***\n")

    for lang in langs:
        print(f"\n{'#'*60}")
        print(f"  Processing: {lang.upper()} ({'TEST' if is_test else 'DEV'})")
        print(f"{'#'*60}")

        # Load documents
        docs = load_documents(lang, args.n, test=is_test)
        print(f"Loaded {len(docs)} {lang} documents from {'test' if is_test else 'dev'} set")

        # Determine optimized program path
        opt_path = args.optimized
        if args.optimized_dir:
            opt_path = f"dir:{args.optimized_dir}"  # Signal to use per-extractor compose

        # Run extraction
        cp = run_extraction(
            model=args.model,
            lang=lang,
            docs=docs,
            resume=args.resume,
            use_cot=not args.no_cot,
            temperature=args.temperature,
            optimized_path=opt_path,
            workers=args.workers,
            few_shot=args.few_shot,
            test=is_test,
            parallel_extractors=args.parallel_extractors,
        )

        # Generate submission
        submission = checkpoint_to_submission(cp, lang, test=is_test)
        sub_path = save_submission(submission, args.model, lang, test=is_test)

        # Score (skip for test mode — no ground truth available)
        if not args.no_score and not is_test:
            # For Italian, use official GT. For English, generate from HF.
            if lang == "it":
                run_scoring(sub_path, lang, DEV_GT_PATH)
            else:
                doc_ids = [d["document_id"] for d in docs]
                gt_path = generate_ground_truth(lang, doc_ids)
                run_scoring(sub_path, lang, gt_path)
        elif is_test:
            print(f"\n  Test mode: scoring skipped. Submission saved at: {sub_path}")


if __name__ == "__main__":
    main()
