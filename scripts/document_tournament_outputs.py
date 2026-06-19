#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

def load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))

def main():
    parser = argparse.ArgumentParser(description="Generate comprehensive documentation for tournament outputs.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Directory for a specific candidate run.")
    parser.add_argument("--candidate-dir", type=Path, required=True, help="Directory for the candidate dataset.")
    args = parser.parse_args()

    adapter_dir = args.run_dir / "adapter"
    train_summary = load_json(adapter_dir / "run_summary.json")
    dataset_card = load_json(args.candidate_dir / "dataset_card.json")
    
    output_path = args.run_dir / "SUMMARY.md"
    lines = [f"# Run Summary: {args.run_dir.name}", ""]

    if dataset_card:
        lines.extend([
            "## Dataset Context",
            f"- **Rows**: {dataset_card.get('row_count')}",
            f"- **Unique Companies**: {dataset_card.get('unique_company_count')}",
            f"- **Why this candidate**: {dataset_card.get('why_this_candidate_exists', 'N/A')}",
            ""
        ])

    if train_summary:
        lines.extend([
            "## Training Details",
            f"- **Start UTC**: {train_summary.get('start_utc')}",
            f"- **End UTC**: {train_summary.get('end_utc')}",
            f"- **Train Loss**: {train_summary.get('train_loss', 'N/A'):.4f}",
            f"- **Peak GPU Mem (Allocated)**: {train_summary.get('peak_gpu_mem_allocated_gb', 'N/A'):.2f} GB",
            ""
        ])

    # Find evaluation subdirectories
    eval_dirs = sorted(list(args.run_dir.glob("eval_*")))
    if eval_dirs:
        lines.append("## Evaluation Results")
        lines.append("")
        for ed in eval_dirs:
            metrics = load_json(ed / "metrics.json")
            if not metrics:
                continue
            
            task_name = ed.name.replace("eval_", "")
            lines.append(f"### Task: {task_name}")
            
            for task_key, task_data in metrics.get("tasks", {}).items():
                for mode, results in task_data.items():
                    adapter_res = results.get("adapter", {})
                    base_res = results.get("base", {})
                    delta = results.get("delta", {})
                    
                    lines.append(f"#### Mode: {mode}")
                    lines.append(f"| Metric | Base | Adapter | Delta |")
                    lines.append(f"| --- | --- | --- | --- |")
                    
                    for m_key in ["accuracy", "macro_f1", "parse_failure_rate"]:
                        if m_key in adapter_res:
                            b_val = base_res.get(m_key, 0.0)
                            a_val = adapter_res.get(m_key, 0.0)
                            d_val = delta.get(m_key, 0.0)
                            lines.append(f"| {m_key} | {b_val:.4f} | {a_val:.4f} | {d_val:+.4f} |")
                    lines.append("")
            
            # Link to detailed eval files
            rel_path = ed.relative_to(args.run_dir)
            lines.append(f"Detailed files: [eval_summary.md]({rel_path}/eval_summary.md), [regression_examples.md]({rel_path}/regression_examples.md)")
            lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Generated {output_path}")

if __name__ == "__main__":
    main()
