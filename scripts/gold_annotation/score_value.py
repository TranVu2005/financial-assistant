"""Score a submission ZIP's answers against the gold-annotated dev benchmark.

This is the real right/wrong check: submission.json inside the ZIP carries
{"id", "answer"} for every question, and dev-benchmark-v1.gold.jsonl carries
the answer read independently from the source report (ADR 0009 A2). Compares
the two directly -- not the coverage-report status, which only says
"answered" vs not.
"""
import json
import sys
import zipfile

zip_path = sys.argv[1]
label = sys.argv[2] if len(sys.argv) > 2 else zip_path

gold = {}
for line in open("data/qa/dev-benchmark-v1.gold.jsonl", encoding="utf-8"):
    if line.strip():
        r = json.loads(line)
        gold[r["vifinqa_id"]] = r

with zipfile.ZipFile(zip_path) as zf:
    json_names = [n for n in zf.namelist() if n.endswith(".json") and "/" not in n]
    assert len(json_names) == 1, json_names
    submission = json.loads(zf.read(json_names[0]).decode("utf-8"))

items = {int(item["id"]): item for item in submission}

# gold["answer"] is already expressed in gold["answer_unit"] (the unit the
# question asks for -- "tỷ đồng" means the stored number IS in billions).
# `compiled.answer`/the submission item's `answer` field are the pipeline's
# own display value in `plan.expected_unit`, which the exporter sets from
# the same "tỷ/triệu/nghìn tỷ đồng" wording. Both sides are therefore
# already in the same display unit -- no rescaling belongs here. (An
# earlier version of this script multiplied gold["answer"] by a VND-per-unit
# factor, which double-converted every answer and made ~all of them look
# wrong -- caught by hand-checking id 4: pipeline gave 444.918, gold's
# recorded answer is 444.918, they match with no scaling at all.)


def close(a: float, b: float, rel: float = 0.01, abs_tol: float = 1e-6) -> bool:
    if abs(a - b) <= abs_tol:
        return True
    denom = max(abs(a), abs(b), 1e-9)
    return abs(a - b) / denom <= rel


annotated = {k: v for k, v in gold.items() if v["status"] == "annotated"}
print(f"=== {label} ===")
print(f"gold-annotated questions: {len(annotated)} / {len(gold)}")
print(f"items in submission.json: {len(items)}")
print()

correct = wrong = not_in_submission = 0
wrong_rows = []
correct_ids = []
for qid, g in sorted(annotated.items()):
    item = items.get(qid)
    if item is None:
        not_in_submission += 1
        continue
    gold_answer = g["answer"]
    also = g.get("also_acceptable") or []
    also_answers = [a["answer"] for a in also]
    got = item["answer"]
    ok = close(got, gold_answer) or any(close(got, a) for a in also_answers)
    if ok:
        correct += 1
        correct_ids.append(qid)
    else:
        wrong += 1
        wrong_rows.append((qid, got, gold_answer, g["answer_unit"], g["question"][:70]))

scored = correct + wrong
print(f"scored (gold present + submitted): {scored}")
print(f"  CORRECT: {correct}")
print(f"  WRONG:   {wrong}")
print(f"  not in submission.json (abstained/backstopped without a real answer path): {not_in_submission}")
if scored:
    print(f"  answer accuracy on scored: {correct / scored:.1%}")
print(f"  answer accuracy on all {len(annotated)} gold-annotated: {correct / len(annotated):.1%}")
print()
print(f"CORRECT ids: {correct_ids}")
print()
print("WRONG:")
for qid, got, want, unit, q in wrong_rows:
    print(f"  {qid:>4}  got={got!r}  want={want!r} [{unit}]  {q}")
