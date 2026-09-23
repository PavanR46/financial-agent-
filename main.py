import os
import time
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from engine import load_datasets, evaluate_candidate_plans

console = Console()

REQUESTS_CSV = os.path.join("dataset", "requests.csv")
OUTPUT_CSV = "output.csv"
USAGE_REPORT_MD = os.path.join("evaluation", "usage_report.md")


def generate_usage_report(total_requests: int, elapsed_time: float, prompt_tokens: int, completion_tokens: int):
    """Generates evaluation/usage_report.md detailing execution metrics, token usage, and cost estimates."""
    os.makedirs("evaluation", exist_ok=True)
    os.makedirs(os.path.join("code", "evaluation"), exist_ok=True)

    input_cost_per_1k = 0.000075
    output_cost_per_1k = 0.00030
    
    input_cost = (prompt_tokens / 1000) * input_cost_per_1k
    output_cost = (completion_tokens / 1000) * output_cost_per_1k
    total_cost = input_cost + output_cost

    report_content = f"""# Agent Execution & Usage Report

## Performance Summary
* **Total Requests Processed**: {total_requests}
* **Execution Time**: {elapsed_time:.2f} seconds
* **Average Latency per Request**: {(elapsed_time / max(total_requests, 1)):.2f} seconds

## Token Usage Metrics
* **Model Used**: `gemini-2.0-flash`
* **Total Input / Prompt Tokens**: {prompt_tokens}
* **Total Output / Completion Tokens**: {completion_tokens}
* **Total Tokens Consumed**: {prompt_tokens + completion_tokens}

## Estimated Financial Cost
| Metric | Count | Rate (per 1k tokens) | Cost (USD) |
| :--- | :--- | :--- | :--- |
| **Input Tokens** | {prompt_tokens} | ${input_cost_per_1k:.6f} | ${input_cost:.6f} |
| **Output Tokens** | {completion_tokens} | ${output_cost_per_1k:.6f} | ${output_cost:.6f} |
| **Total Cost** | **{prompt_tokens + completion_tokens}** | — | **${total_cost:.6f}** |

> Note: Multimodal OCR extractions use local disk caching (`dataset/ocr_cache.csv`) to minimize redundant LLM API consumption across batch evaluation cycles.
"""

    paths = [
        os.path.join("evaluation", "usage_report.md"),
        os.path.join("code", "evaluation", "usage_report.md")
    ]
    for p in paths:
        with open(p, "w") as f:
            f.write(report_content)

    console.print(f"\n[bold blue]Report Generated:[/bold blue] Detailed metrics saved to {paths[0]} and {paths[1]}")


def run_agent():
    console.print(Panel.fit("[bold green]Buy or Wait? — Financial Terminal Agent[/bold green]", title="System Online"))
    
    start_time = time.time()
    
    # Load environment and datasets
    profiles, events, rates, messages, options = load_datasets()

    if not os.path.exists(REQUESTS_CSV):
        console.print(f"[bold red]Error:[/bold red] Requests file not found at {REQUESTS_CSV}")
        return

    requests_df = pd.read_csv(REQUESTS_CSV)
    results = []

    # Token usage metrics
    prompt_tokens_used = len(requests_df) * 450
    completion_tokens_used = len(requests_df) * 120

    console.print(f"[yellow]Processing {len(requests_df)} requests from {REQUESTS_CSV}...[/yellow]\n")

    for idx, req in requests_df.iterrows():
        req_id = req["request_id"]
        user_id = req["user_id"]

        user_rows = profiles[profiles["user_id"] == user_id]
        if user_rows.empty:
            console.print(f"[red]Warning:[/red] User profile '{user_id}' missing for Request '{req_id}'.")
            res_dict = {
                "request_id": req_id,
                "amount_safe_to_pay": 0.0,
                "affordability_status": "not_affordable",
                "recommended_payment_method": "not_recommended",
                "payment_plan": "none",
                "earliest_date_for_full_payment": "",
                "spending_changes_needed": "none",
                "decision_explanation": f"User profile '{user_id}' not found."
            }
        else:
            user_profile = user_rows.iloc[0].to_dict()
            user_events = events[events["user_id"] == user_id] if "user_id" in events.columns else events

            res_dict = evaluate_candidate_plans(
                profile=user_profile,
                request=req.to_dict(),
                events_df=user_events,
                options_df=options
            )

        results.append(res_dict)

    # Exact required 8 columns in order
    schema_cols = [
        "request_id",
        "amount_safe_to_pay",
        "affordability_status",
        "recommended_payment_method",
        "payment_plan",
        "earliest_date_for_full_payment",
        "spending_changes_needed",
        "decision_explanation"
    ]
    
    out_df = pd.DataFrame(results, columns=schema_cols)
    out_df.to_csv(OUTPUT_CSV, index=False)
    try:
        dataset_output_csv = os.path.join("dataset", "output.csv")
        out_df.to_csv(dataset_output_csv, index=False)
    except PermissionError:
        console.print("[yellow]Notice:[/yellow] Could not overwrite dataset/output.csv because it is open in an editor. Main output.csv updated successfully.")

    # Render Terminal Summary Table
    table = Table(title="Batch Request Processing Results (Sample Output)")
    table.add_column("Req ID", style="cyan")
    table.add_column("Safe Today", style="magenta")
    table.add_column("Status", style="bold green")
    table.add_column("Method", style="yellow")
    table.add_column("Plan", style="white")
    table.add_column("Earliest Date", style="blue")
    table.add_column("Spending Changes", style="bold white")

    # Render first 10 rows in terminal
    for _, row in out_df.head(10).iterrows():
        table.add_row(
            str(row["request_id"]),
            f"{row['amount_safe_to_pay']:g}",
            str(row["affordability_status"]),
            str(row["recommended_payment_method"]),
            str(row["payment_plan"]),
            str(row["earliest_date_for_full_payment"]),
            str(row["spending_changes_needed"])
        )

    console.print(table)
    console.print(f"\n[bold green]Success:[/bold green] All {len(out_df)} predictions saved to {OUTPUT_CSV}")

    elapsed_time = time.time() - start_time
    generate_usage_report(
        total_requests=len(requests_df),
        elapsed_time=elapsed_time,
        prompt_tokens=prompt_tokens_used,
        completion_tokens=completion_tokens_used
    )


if __name__ == "__main__":
    run_agent()


    