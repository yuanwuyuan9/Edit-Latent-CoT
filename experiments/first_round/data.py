"""Dataset adaptation and answer scoring; no model dependencies."""
from __future__ import annotations

import hashlib
import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

EVALUATOR_VERSION = "first-round-v1"
NUMBER = re.compile(r"(?<![\w.])[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![\w.])")


def numeric_answer(text: str) -> str | None:
    # Only call this on the generated continuation, never the prompt.
    tail = text.rsplit("###", 1)[-1]
    values = NUMBER.findall(tail)
    if not values:
        return None
    try:
        value = Decimal(values[-1].replace(",", ""))
        return format(value.normalize(), "f")
    except InvalidOperation:
        return None


def prosqa_options(question: str) -> list[str]:
    match = re.search(r"Is\s+\w+\s+a\s+(\w+)\s+or\s+(\w+)\?\s*$", question)
    if not match:
        raise ValueError("ProsQA question must end with two candidate predicates")
    return [value.lower() for value in match.groups()]


def prosqa_answer(text: str, options: list[str]) -> str | None:
    tail = text.rsplit("###", 1)[-1].lower()
    found = [option for option in options if re.search(r"\b" + re.escape(option) + r"\b", tail)]
    # Mentioning both options is ambiguous; do not guess the last one.
    return found[0] if len(found) == 1 else None


def load_samples(path: Path, dataset: str, split: str) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    rows = [json.loads(line) for line in text.splitlines() if line.strip()] if path.suffix == ".jsonl" else json.loads(text)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Dataset must be a nonempty JSON array or JSONL")
    samples = []
    for index, row in enumerate(rows):
        question, answer = row["question"], row["answer"]
        if not isinstance(question, str) or not isinstance(answer, str):
            raise ValueError(f"Invalid question/answer at row {index}")
        options = None
        if dataset == "gsm8k":
            if "####" not in answer:
                raise ValueError(f"GSM8K gold answer has no #### delimiter at row {index}")
            scalar = answer.rsplit("####", 1)[-1].strip()
            if NUMBER.fullmatch(scalar) is None:
                raise ValueError(f"GSM8K gold is not a scalar number at row {index}")
            gold = numeric_answer(scalar)
        elif dataset == "prosqa":
            options = prosqa_options(question)
            gold = prosqa_answer(answer, options)
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")
        if gold is None:
            raise ValueError(f"Cannot parse gold answer at row {index}")
        question_hash = hashlib.sha256(question.encode()).hexdigest()[:12]
        samples.append({
            "sample_id": f"{dataset}/{split}/{index}:{question_hash}",
            "index": index, "question": question, "gold_answer": gold,
            "reference_answer": answer, "reference_steps": row.get("steps"),
            "options": options,
        })
    return samples


def score(continuation: str, sample: dict, dataset: str) -> dict:
    answer = numeric_answer(continuation) if dataset == "gsm8k" else prosqa_answer(continuation, sample["options"])
    return {
        "predicted_answer": answer,
        "parse_status": "ok" if answer is not None else "failed",
        "correct": answer is not None and answer == sample["gold_answer"],
        "evaluator_version": EVALUATOR_VERSION,
    }


def summarize(records: list[dict]) -> dict:
    successes = [row for row in records if row["status"] == "ok"]
    correct = sum(row["correct"] for row in successes)
    result = {
        "record_count": len(records), "completed": len(successes),
        "execution_failed": len(records) - len(successes),
        "parse_failed": sum(row["parse_status"] == "failed" for row in successes),
        "parsed_wrong": sum(row["parse_status"] == "ok" and not row["correct"] for row in successes),
        "accuracy": {"numerator": correct, "denominator": len(successes),
                     "value": correct / len(successes) if successes else None},
    }
    paired = [row for row in successes if "baseline_correct" in row]
    if paired:
        eligible = [row for row in paired if row.get("baseline_parse_status", "ok") == "ok"]
        result["paired_baseline_parse_failed"] = len(paired) - len(eligible)
        wrong = [row for row in eligible if not row["baseline_correct"]]
        right = [row for row in eligible if row["baseline_correct"]]
        result["wrong_to_correct"] = {"numerator": sum(row["correct"] for row in wrong), "denominator": len(wrong)}
        result["correct_to_wrong"] = {"numerator": sum(not row["correct"] for row in right), "denominator": len(right)}
        result["answer_changed"] = sum(row["predicted_answer"] != row["baseline_answer"] for row in paired)
        result["text_changed"] = sum(row["continuation"] != row["baseline_continuation"] for row in paired)
    checks = [row for row in successes if "passed" in row]
    if checks:
        result["checks_passed"] = sum(row["passed"] for row in checks)
        result["checks_total"] = len(checks)
    return result
