"""
Evaluate Engram memory system on LoCoMo benchmark.

This uses the official LoCoMo evaluation methodology with Engram
as the memory/retrieval backend.

Usage:
    python benchmark/evaluate_engram.py \
        --data-file benchmark/locomo10.json \
        --out-file benchmark/results/engram_results.json \
        --model gemini-2.0-flash-lite
"""

import sys
from pathlib import Path
import os
import json
import argparse
from tqdm import tqdm

# Add engram root to path for imports
# benchmark/evaluate_engram.py -> benchmark -> engram
ENGRAM_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ENGRAM_ROOT))

# Add locomo-official to path
# benchmark/evaluate_engram.py -> benchmark -> locomo-official
LOCOMO_ROOT = Path(__file__).parent / "locomo-official"
sys.path.insert(0, str(LOCOMO_ROOT))

from task_eval.evaluation import eval_question_answering
from task_eval.evaluation_stats import analyze_aggr_acc
from task_eval.engram_utils import (
    init_engram_memory, get_engram_answers, shutdown_engram,
    init_baseline, get_baseline_answers, shutdown_baseline
)
from token_tracker import tracker


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Engram on LoCoMo")
    parser.add_argument('--data-file', type=str, default='benchmark/locomo-official/data/locomo10.json', help="Path to locomo10.json")
    parser.add_argument('--out-file', type=str, required=True, help="Output file for results")
    parser.add_argument('--model', type=str, default='gemini-2.0-flash-lite', help="Model to use")
    parser.add_argument('--max-conversations', type=int, default=None, help="Limit conversations (for testing)")
    parser.add_argument('--max-qa', type=int, default=None, help="Limit QA per conversation")
    parser.add_argument('--overwrite', action='store_true', help="Overwrite existing predictions")
    parser.add_argument('--memory-dir', type=str, default='benchmark/benchmark_memories', help="Memory storage directory")
    parser.add_argument('--baseline', action='store_true', help="Run baseline (vanilla Gemini, no Engram)")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Reset token tracker for fresh benchmark run
    tracker.reset()
    tracker.set_log_path(os.path.join(os.path.dirname(args.out_file), 'token_usage.jsonl'))
    
    mode = "baseline" if args.baseline else "engram"
    
    print("=" * 60)
    if args.baseline:
        print("📊 BASELINE LoCoMo Benchmark (Vanilla Gemini)")
    else:
        print("🧠 ENGRAM LoCoMo Benchmark")
    print("=" * 60)
    print(f"Mode: {mode}")
    print(f"Model: {args.model}")
    print(f"Data: {args.data_file}")
    print(f"Output: {args.out_file}")
    
    # Load dataset
    samples = json.load(open(args.data_file))
    if args.max_conversations:
        samples = samples[:args.max_conversations]
    
    print(f"Evaluating {len(samples)} conversations")
    
    # Model key for results
    model_key = f"{mode}_{args.model}"
    prediction_key = f"{model_key}_prediction"
    
    # Load existing results if any
    if os.path.exists(args.out_file):
        out_samples = {d['sample_id']: d for d in json.load(open(args.out_file))}
    else:
        out_samples = {}
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.out_file) or '.', exist_ok=True)
    os.makedirs(args.memory_dir, exist_ok=True)
    
    # Process each conversation
    for data in tqdm(samples, desc="Processing conversations"):
        sample_id = data['sample_id']
        print(f"\n📖 Processing conversation: {sample_id}")
        
        # Prepare output data
        out_data = {'sample_id': sample_id}
        if sample_id in out_samples:
            out_data['qa'] = out_samples[sample_id]['qa'].copy()
        else:
            out_data['qa'] = data['qa'].copy()
        
        # Limit QA if requested
        if args.max_qa:
            out_data['qa'] = out_data['qa'][:args.max_qa]
            # Also limit input QA to match
            limited_data = data.copy()
            limited_data['qa'] = data['qa'][:args.max_qa]
        else:
            limited_data = data
        
def calculate_and_save_stats(answers, out_samples, sample_id, model_key, prediction_key, out_file):
    """
    Shared function to calculate F1 scores, update results, and save them.
    """
    # Evaluate using official LoCoMo metrics
    exact_matches, lengths, recall = eval_question_answering(
        answers['qa'], 
        eval_key=prediction_key
    )
    
    # Store F1 scores
    for i in range(len(answers['qa'])):
        answers['qa'][i][model_key + '_f1'] = round(exact_matches[i], 3)
    
    out_samples[sample_id] = answers
    
    # Save intermediate results
    with open(out_file, 'w') as f:
        json.dump(list(out_samples.values()), f, indent=2)

def main():
    args = parse_args()
    
    # Reset token tracker for fresh benchmark run
    tracker.reset()
    tracker.set_log_path(os.path.join(os.path.dirname(args.out_file), 'token_usage.jsonl'))
    
    mode = "baseline" if args.baseline else "engram"
    
    print("=" * 60)
    if args.baseline:
        print("📊 BASELINE LoCoMo Benchmark (Vanilla Gemini)")
    else:
        print("🧠 ENGRAM LoCoMo Benchmark")
    print("=" * 60)
    print(f"Mode: {mode}")
    print(f"Model: {args.model}")
    print(f"Data: {args.data_file}")
    print(f"Output: {args.out_file}")
    
    # Load dataset
    samples = json.load(open(args.data_file))
    if args.max_conversations:
        samples = samples[:args.max_conversations]
    
    print(f"Evaluating {len(samples)} conversations")
    
    # Model key for results
    model_key = f"{mode}_{args.model}"
    prediction_key = f"{model_key}_prediction"
    
    # Load existing results if any
    if os.path.exists(args.out_file):
        out_samples = {d['sample_id']: d for d in json.load(open(args.out_file))}
    else:
        out_samples = {}
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.out_file) or '.', exist_ok=True)
    os.makedirs(args.memory_dir, exist_ok=True)
    
    # Process each conversation
    for data in tqdm(samples, desc="Processing conversations"):
        sample_id = data['sample_id']
        print(f"\n📖 Processing conversation: {sample_id}")
        
        # Prepare output data
        out_data = {'sample_id': sample_id}
        if sample_id in out_samples:
            out_data['qa'] = out_samples[sample_id]['qa'].copy()
        else:
            out_data['qa'] = data['qa'].copy()
        
        # Limit QA if requested
        if args.max_qa:
            out_data['qa'] = out_data['qa'][:args.max_qa]
            # Also limit input QA to match
            limited_data = data.copy()
            limited_data['qa'] = data['qa'][:args.max_qa]
        else:
            limited_data = data
        
        if args.baseline:
            # BASELINE: Vanilla Gemini with full conversation context
            baseline = init_baseline(data['conversation'], model=args.model)
            try:
                # Generate answers
                answers = get_baseline_answers(baseline, limited_data, out_data, prediction_key, args)
                calculate_and_save_stats(answers, out_samples, sample_id, model_key, prediction_key, args.out_file)
            finally:
                shutdown_baseline(baseline)
        else:
            # ENGRAM: Memory-augmented
            memory_path = os.path.join(args.memory_dir, sample_id)
            proxy = init_engram_memory(
                data['conversation'], 
                memory_path, 
                model=args.model
            )
            try:
                # Generate answers
                answers = get_engram_answers(proxy, limited_data, out_data, prediction_key, args)
                calculate_and_save_stats(answers, out_samples, sample_id, model_key, prediction_key, args.out_file)
            finally:
                shutdown_engram(proxy)
    
    # Generate statistics

    stats_file = args.out_file.replace('.json', '_stats.json')
    analyze_aggr_acc(
        args.data_file, 
        args.out_file, 
        stats_file,
        model_key, 
        model_key + '_f1'
    )
    
    print("\n" + "=" * 60)
    print(f"✅ Benchmark complete! ({mode.upper()})")
    print(f"Results: {args.out_file}")
    print(f"Stats: {stats_file}")
    print("=" * 60)
    
    # Print token usage stats
    print()
    print(tracker.format_stats())
    
    # Save token stats to results
    token_stats = tracker.get_stats()
    token_stats_file = args.out_file.replace('.json', '_tokens.json')
    with open(token_stats_file, 'w') as f:
        json.dump(token_stats, f, indent=2)
    print(f"\n💰 Token stats saved to: {token_stats_file}")


if __name__ == "__main__":
    main()

