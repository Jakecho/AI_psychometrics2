"""Pre-response fields shared by preview and blinded review."""

from hashlib import sha256

import core


def facts(case, stage_id):
    rows = []
    for fact in core.visible_facts(case, stage_id):
        row = {"fact_id": fact["id"], "label": fact.get("name", ""), "value": fact.get("value", ""),
               **{k: fact[k] for k in ("unit", "status") if k in fact}}
        status = str(fact.get("status", "")).casefold().replace("_", " ")
        if status in {"unknown", "unassessed", "not assessed", "undocumented"}:
            row["value"] = None
        elif status == "absent":
            row["value"] = "absent"
        rows.append(row)
    return rows


def fact_text(fact):
    value = fact.get("status", "unknown") if fact["value"] is None else fact["value"]
    return f"{fact['label']}: {value}" + (f" {fact['unit']}" if fact.get("unit") else "")


def disclosures(case):
    return [
        {"stage_id": stage["id"], "stage_title": stage["title"],
         "new_information": stage["new_information"], "elapsed": stage.get("elapsed", ""),
         "transition": case["stages"][index - 1]["transition"] if index else ""}
        for index, stage in enumerate(case["stages"])
    ]


def option_token(item, option_id):
    return "choice_" + sha256(f"{item['id']}\0{option_id}".encode()).hexdigest()[:24]


def item_view(case, item, *, opaque=False):
    view = {
        "item_title": item["title"], "stem": item["stem"],
        "lead_in": item["lead_in"] or "No lead-in entered.",
        "selection_instruction": item["selection_instruction"],
        "options": [{"option_id": option_token(item, option["id"]) if opaque else option["id"],
                     "text": option["text"] or ("(blank option)" if item["format"] == "SBA" else f"Blank option {i}")}
                    for i, option in enumerate(item["options"], 1)],
    }
    if case["mode"] == "appraisal":
        view["preliminary_recommendation"] = item["recommendation"] or "No preliminary recommendation entered."
    return view


def stage_view(case, stage_id):
    return {
        "shared_vignette": case["patient"] or "_Shared vignette has not been written._",
        "visible_disclosures": disclosures(case)[:[s["id"] for s in case["stages"]].index(stage_id) + 1],
    }


def delivery_policy(case):
    return {
        "response_lock": "An earlier stage locks when the learner advances. Submitted responses can be revised within the current stage.",
        "item_navigation": "All items in the active stage can be answered in any order. Earlier stages remain visible.",
        "feedback_timing": "Preview reveals an item's key and rationales immediately upon submission while current-stage responses remain editable.",
    }


def sequence(case):
    return [
        disclosure | {
            "facts": facts(case, disclosure["stage_id"]),
            "items": [{"item_id": item["id"]} | item_view(case, item, opaque=True)
                      for item in case["items"] if item["stage_id"] == disclosure["stage_id"]],
        }
        for disclosure in disclosures(case)
    ]
