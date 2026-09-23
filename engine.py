import os
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from ocr import resolve_missing_image_amounts

DATASET_DIR = "dataset"


def fmt_amt(val: float) -> str:
    """Formats currency amounts cleanly without scientific notation."""
    if pd.isna(val):
        return "0"
    val = float(val)
    if val.is_integer():
        return f"{int(val)}"
    return f"{val:.2f}".rstrip('0').rstrip('.')


def load_datasets():
    """Loads all financial CSV datasets into Pandas DataFrames."""
    profiles = pd.read_csv(os.path.join(DATASET_DIR, "financial_profiles.csv"))
    events = pd.read_csv(os.path.join(DATASET_DIR, "financial_events.csv"))
    rates = pd.read_csv(os.path.join(DATASET_DIR, "exchange_rates.csv"))
    messages = pd.read_csv(os.path.join(DATASET_DIR, "messages.csv"))
    options = pd.read_csv(os.path.join(DATASET_DIR, "request_payment_options.csv"))
    
    # Merge OCR resolved amounts into financial events
    try:
        resolved_ocr = resolve_missing_image_amounts()
        if not resolved_ocr.empty and "related_event_id" in resolved_ocr.columns and "amount" in resolved_ocr.columns:
            ocr_map = dict(zip(resolved_ocr["related_event_id"], resolved_ocr["amount"]))
            events["amount"] = events.apply(
                lambda r: ocr_map.get(r["event_id"], r["amount"]), axis=1
            )
    except Exception as e:
        print(f"Warning: OCR integration skipped due to: {e}")

    return profiles, events, rates, messages, options


def convert_currency(amount: float, from_curr: str, to_curr: str, event_date: str, rates_df: pd.DataFrame) -> float:
    """Converts amount from from_curr to to_curr using exchange rates on event_date."""
    if from_curr == to_curr or pd.isna(amount) or amount == 0:
        return float(amount) if pd.notna(amount) else 0.0

    rate_row = rates_df[
        (rates_df["date"] == event_date) & 
        (rates_df["from_currency"] == from_curr) & 
        (rates_df["to_currency"] == to_curr)
    ]

    if not rate_row.empty:
        return amount * float(rate_row.iloc[0]["rate"])

    inv_rate_row = rates_df[
        (rates_df["date"] == event_date) & 
        (rates_df["from_currency"] == to_curr) & 
        (rates_df["to_currency"] == from_curr)
    ]
    if not inv_rate_row.empty:
        return amount / float(inv_rate_row.iloc[0]["rate"])

    pair_rates = rates_df[
        (rates_df["from_currency"] == from_curr) & 
        (rates_df["to_currency"] == to_curr)
    ]
    if not pair_rates.empty:
        return amount * float(pair_rates.iloc[-1]["rate"])

    return amount


def normalize_financial_events(events_df: pd.DataFrame, messages_df: pd.DataFrame, rates_df: pd.DataFrame, home_currency: str) -> pd.DataFrame:
    """Filters invalid transactions, applies message amendments, and normalizes amounts into home currency."""
    df = events_df.copy()
    df = df.drop_duplicates(subset=["event_id"])

    valid_statuses = ["completed", "settled", "posted", "processed"]
    if "status" in df.columns:
        df = df[df["status"].str.lower().isin(valid_statuses)]

    if "event_type" in df.columns:
        df = df[~df["event_type"].str.lower().isin(["unrealized_investment", "pending_credit"])]

    if not messages_df.empty:
        cancelled_ids = set()
        amendments = {}

        for _, msg in messages_df.iterrows():
            action = str(msg.get("action", "")).lower()
            target_id = msg.get("target_event_id")

            if action in ["cancel", "void", "delete"]:
                cancelled_ids.add(target_id)
            elif action == "amend" and pd.notna(msg.get("new_amount")):
                amendments[target_id] = float(msg.get("new_amount"))

        df = df[~df["event_id"].isin(cancelled_ids)]
        for event_id, new_amt in amendments.items():
            df.loc[df["event_id"] == event_id, "amount"] = new_amt

    normalized_amounts = []
    for _, row in df.iterrows():
        converted = convert_currency(
            amount=row["amount"],
            from_curr=row["currency"],
            to_curr=home_currency,
            event_date=row["event_date"],
            rates_df=rates_df
        )
        normalized_amounts.append(converted)

    df["normalized_amount"] = normalized_amounts
    return df


def build_daily_event_map(events_df: pd.DataFrame) -> Dict[str, float]:
    """Pre-aggregates net cash impact by date string for fast 90-day balance simulation."""
    event_map = {}
    for _, row in events_df.iterrows():
        d_str = str(row["event_date"])
        amt = float(row.get("normalized_amount", row.get("amount", 0.0)))
        cat = str(row.get("category", "")).lower()
        impact = amt if cat == "income" else -amt
        event_map[d_str] = event_map.get(d_str, 0.0) + impact
    return event_map


def generate_90_day_balance_walk(
    start_date_str: str,
    initial_balance: float,
    daily_event_map: Dict[str, float],
    extra_payments: Optional[List[Dict[str, Any]]] = None,
    adjustments_map: Optional[Dict[str, float]] = None
) -> List[float]:
    """Fast simulation of daily end-of-day balances over a 90-day window."""
    extra_map = {}
    if extra_payments:
        for p in extra_payments:
            d_str = p["date"]
            extra_map[d_str] = extra_map.get(d_str, 0.0) + p["amount"]

    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    current_bal = initial_balance
    daily_balances = []

    for d in range(91):
        curr_dt = start_date + timedelta(days=d)
        curr_str = curr_dt.strftime("%Y-%m-%d")
        
        day_impact = daily_event_map.get(curr_str, 0.0)

        if extra_map and curr_str in extra_map:
            day_impact -= extra_map[curr_str]

        if adjustments_map and curr_str in adjustments_map:
            day_impact += adjustments_map[curr_str]

        current_bal += day_impact
        daily_balances.append(current_bal)

    return daily_balances


def is_balance_safe(daily_balances: List[float], min_balance: float) -> bool:
    """Checks whether daily balance remains >= minimum balance threshold."""
    return all(bal >= min_balance - 1e-4 for bal in daily_balances)


def compute_amount_safe_to_pay(
    start_date_str: str,
    curr_bal: float,
    min_bal: float,
    req_amount: float,
    daily_event_map: Dict[str, float]
) -> float:
    """Calculates max safe amount to pay today without breaking 90-day min balance requirement."""
    baseline_walk = generate_90_day_balance_walk(start_date_str, curr_bal, daily_event_map)
    min_projected = min(baseline_walk)
    safe = max(0.0, min_projected - min_bal)
    return min(req_amount, round(safe, 2))


def find_earliest_date_for_full_payment(
    start_date_str: str,
    curr_bal: float,
    min_bal: float,
    req_amount: float,
    daily_event_map: Dict[str, float]
) -> str:
    """Finds the first date within 90 days when full payment is safe as a single payment."""
    start_dt = datetime.strptime(start_date_str, "%Y-%m-%d")
    for delay in range(91):
        test_dt = start_dt + timedelta(days=delay)
        test_str = test_dt.strftime("%Y-%m-%d")
        
        walk = generate_90_day_balance_walk(
            start_date_str,
            curr_bal,
            daily_event_map,
            extra_payments=[{"date": test_str, "amount": req_amount}]
        )
        if is_balance_safe(walk, min_bal):
            return test_str
    return ""


def evaluate_candidate_plans(
    profile: dict,
    request: dict,
    events_df: pd.DataFrame,
    options_df: pd.DataFrame
) -> dict:
    """Evaluates candidates dynamically and ranks them strictly by §6.3 rules."""
    user_id = profile["user_id"]
    req_id = request["request_id"]
    min_bal = float(profile["minimum_balance_to_keep"])
    curr_bal = float(profile["current_available_balance"])
    req_amount = float(request["requested_amount"])
    req_date = request["request_date"]
    home_currency = profile["home_currency"]
    desired_completion = request.get("desired_completion_date", req_date)
    allows_partial = str(request.get("allows_partial_payment", "")).lower() in ["true", "1", "yes"]

    pref_methods = [m.strip().lower() for m in str(profile.get("payment_methods_user_will_consider", "")).split("|") if m.strip()]
    max_months_val = profile.get("max_installment_months")
    max_months = int(max_months_val) if pd.notna(max_months_val) and str(max_months_val).strip().isdigit() else 99

    daily_event_map = build_daily_event_map(events_df)
    amount_safe_today = compute_amount_safe_to_pay(req_date, curr_bal, min_bal, req_amount, daily_event_map)
    earliest_full_date = find_earliest_date_for_full_payment(req_date, curr_bal, min_bal, req_amount, daily_event_map)

    candidates = []

    # --- Option A: Full Payment Today ---
    if "full_payment" in pref_methods:
        full_walk = generate_90_day_balance_walk(
            req_date, curr_bal, daily_event_map,
            extra_payments=[{"date": req_date, "amount": req_amount}]
        )
        if is_balance_safe(full_walk, min_bal):
            candidates.append({
                "affordability_status": "affordable_now",
                "recommended_payment_method": "full_payment",
                "payment_plan": f"{req_date}:{fmt_amt(req_amount)}",
                "earliest_date_for_full_payment": req_date,
                "spending_changes_needed": "none",
                "completion_date": req_date,
                "spending_changes": False,
                "total_amount_paid": req_amount,
                "start_date": req_date,
                "num_payments": 1,
                "option_id": "0_full_now",
                "explanation": f"Pay {home_currency} {fmt_amt(req_amount)} today. This keeps the {home_currency} {fmt_amt(min_bal)} minimum available over the next 90 days."
            })

    # --- Option B: Installments ---
    if "installments" in pref_methods and not options_df.empty:
        req_options = options_df[options_df["request_id"] == req_id]
        for _, opt in req_options.iterrows():
            opt_id = str(opt["payment_option_id"])
            opt_method = str(opt.get("payment_method", "")).lower()
            num_payments = int(opt.get("number_of_payments", 1))
            pmt_amt = float(opt.get("payment_amount", 0.0))
            first_date = str(opt.get("first_payment_date", req_date))
            freq_days = int(opt.get("payment_frequency_days", 30)) if pd.notna(opt.get("payment_frequency_days")) else 30
            total_payable = float(opt.get("total_payable_amount", pmt_amt * num_payments))

            if opt_method == "installments" and num_payments <= max_months:
                start_dt = datetime.strptime(first_date, "%Y-%m-%d")
                inst_payments = []
                plan_parts = []
                for i in range(num_payments):
                    p_dt = start_dt + timedelta(days=i * freq_days)
                    p_str = p_dt.strftime("%Y-%m-%d")
                    inst_payments.append({"date": p_str, "amount": pmt_amt})
                    plan_parts.append(f"{p_str}:{fmt_amt(pmt_amt)}")

                inst_walk = generate_90_day_balance_walk(
                    req_date, curr_bal, daily_event_map, extra_payments=inst_payments
                )

                if is_balance_safe(inst_walk, min_bal):
                    last_date = inst_payments[-1]["date"]
                    candidates.append({
                        "affordability_status": "affordable_with_plan",
                        "recommended_payment_method": "installments",
                        "payment_plan": "|".join(plan_parts),
                        "earliest_date_for_full_payment": earliest_full_date,
                        "spending_changes_needed": "none",
                        "completion_date": last_date,
                        "spending_changes": False,
                        "total_amount_paid": total_payable,
                        "start_date": first_date,
                        "num_payments": num_payments,
                        "option_id": opt_id,
                        "explanation": f"Use {num_payments} installments of {home_currency} {fmt_amt(pmt_amt)}, starting {first_date}."
                    })

    # --- Option C: Partial Payment ---
    if "partial_payment" in pref_methods and allows_partial:
        if 0 < amount_safe_today < req_amount and earliest_full_date and earliest_full_date <= desired_completion:
            remainder = round(req_amount - amount_safe_today, 2)
            part_payments = [
                {"date": req_date, "amount": amount_safe_today},
                {"date": earliest_full_date, "amount": remainder}
            ]
            part_walk = generate_90_day_balance_walk(
                req_date, curr_bal, daily_event_map, extra_payments=part_payments
            )
            if is_balance_safe(part_walk, min_bal):
                candidates.append({
                    "affordability_status": "affordable_with_plan",
                    "recommended_payment_method": "partial_payment",
                    "payment_plan": f"{req_date}:{fmt_amt(amount_safe_today)}|{earliest_full_date}:{fmt_amt(remainder)}",
                    "earliest_date_for_full_payment": earliest_full_date,
                    "spending_changes_needed": "none",
                    "completion_date": earliest_full_date,
                    "spending_changes": False,
                    "total_amount_paid": req_amount,
                    "start_date": req_date,
                    "num_payments": 2,
                    "option_id": "0_partial",
                    "explanation": f"Pay {home_currency} {fmt_amt(amount_safe_today)} today and remaining {home_currency} {fmt_amt(remainder)} on {earliest_full_date}."
                })

    # --- Option D: Wait (Full Payment Later) ---
    if "full_payment" in pref_methods and earliest_full_date and earliest_full_date > req_date:
        wait_walk = generate_90_day_balance_walk(
            req_date, curr_bal, daily_event_map,
            extra_payments=[{"date": earliest_full_date, "amount": req_amount}]
        )
        if is_balance_safe(wait_walk, min_bal):
            candidates.append({
                "affordability_status": "affordable_later",
                "recommended_payment_method": "wait",
                "payment_plan": f"{earliest_full_date}:{fmt_amt(req_amount)}",
                "earliest_date_for_full_payment": earliest_full_date,
                "spending_changes_needed": "none",
                "completion_date": earliest_full_date,
                "spending_changes": False,
                "total_amount_paid": req_amount,
                "start_date": earliest_full_date,
                "num_payments": 1,
                "option_id": "0_wait",
                "explanation": f"Pay {home_currency} {fmt_amt(req_amount)} in full on {earliest_full_date}. Paying earlier would take the balance below minimum."
            })

    # --- Option E: Flexible Spending Adjustments ---
    if not candidates:
        prot_cats = set(str(profile.get("expense_categories_to_protect", "")).split("|"))
        stop_cats = set(str(profile.get("expense_categories_user_is_willing_to_stop", "")).split("|"))
        red_cats = set(str(profile.get("expense_categories_user_is_willing_to_reduce", "")).split("|"))

        user_events = events_df[events_df["user_id"] == user_id].copy() if "user_id" in events_df.columns else events_df.copy()
        
        adjustments_map = {}
        spending_actions = []

        for _, ev in user_events.iterrows():
            cat = str(ev.get("category", ""))
            e_id = ev.get("event_id")
            amt = float(ev.get("normalized_amount", ev.get("amount", 0.0)))
            ev_date = str(ev.get("event_date", ""))

            if cat in prot_cats or not e_id:
                continue

            if cat in stop_cats:
                adjustments_map[ev_date] = adjustments_map.get(ev_date, 0.0) + amt
                spending_actions.append(f"stop:{e_id}")
            elif cat in red_cats:
                savings = amt * 0.5
                adjustments_map[ev_date] = adjustments_map.get(ev_date, 0.0) + savings
                spending_actions.append(f"reduce_to:{e_id}:{fmt_amt(amt*0.5)}")

            if len(spending_actions) >= 3:
                break

        if adjustments_map:
            flex_walk = generate_90_day_balance_walk(
                req_date, curr_bal, daily_event_map,
                extra_payments=[{"date": req_date, "amount": req_amount}],
                adjustments_map=adjustments_map
            )
            if is_balance_safe(flex_walk, min_bal):
                sp_str = "|".join(spending_actions)
                candidates.append({
                    "affordability_status": "affordable_with_plan",
                    "recommended_payment_method": "full_payment",
                    "payment_plan": f"{req_date}:{fmt_amt(req_amount)}",
                    "earliest_date_for_full_payment": earliest_full_date or req_date,
                    "spending_changes_needed": sp_str,
                    "completion_date": req_date,
                    "spending_changes": True,
                    "total_amount_paid": req_amount,
                    "start_date": req_date,
                    "num_payments": 1,
                    "option_id": "0_flex",
                    "explanation": f"Apply spending adjustments ({sp_str}), then pay {home_currency} {fmt_amt(req_amount)} today."
                })

    # --- Select & Rank Best Plan ---
    if candidates:
        best = rank_and_select_best_plan(candidates, desired_completion)
        return {
            "request_id": req_id,
            "amount_safe_to_pay": amount_safe_today,
            "affordability_status": best["affordability_status"],
            "recommended_payment_method": best["recommended_payment_method"],
            "payment_plan": best["payment_plan"],
            "earliest_date_for_full_payment": best["earliest_date_for_full_payment"],
            "spending_changes_needed": best["spending_changes_needed"],
            "decision_explanation": best["explanation"]
        }

    # --- Fallback: Not Affordable ---
    return {
        "request_id": req_id,
        "amount_safe_to_pay": amount_safe_today,
        "affordability_status": "not_affordable",
        "recommended_payment_method": "not_recommended",
        "payment_plan": "none",
        "earliest_date_for_full_payment": earliest_full_date,
        "spending_changes_needed": "none",
        "decision_explanation": f"Do not make this payment by {desired_completion}. None of the available options keeps the {home_currency} {fmt_amt(min_bal)} minimum protected."
    }


def rank_and_select_best_plan(candidates: List[dict], desired_completion_date: str) -> dict:
    """Ranks candidate plans according to strict §6.3 priority rules."""
    def sorting_key(c):
        meets_completion = c["completion_date"] <= desired_completion_date
        return (
            0 if meets_completion else 1,            # 1. Complete full request by desired date
            0 if not c["spending_changes"] else 1,    # 2. Require no spending changes
            c["total_amount_paid"],                 # 3. Minimize total amount paid
            c["start_date"],                        # 4. Start payment earlier
            c["num_payments"],                      # 5. Use fewer payments
            c["option_id"]                          # 6. Tie-breaker: lowest payment_option_id
        )

    candidates.sort(key=sorting_key)
    return candidates[0]