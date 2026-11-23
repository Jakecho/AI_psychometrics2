"""Exercise the FIT authoring views through the local and hosted Streamlit apps."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import ai_adapter
import core
import specialist_workflow as workflow
import storage
from test_specialist_workflow import fit_case, option


APP = Path(__file__).resolve().parent
COLONOSCOPY = "Arrange diagnostic colonoscopy"
ANNUAL = "Repeat screening FIT in one year"
NEW_SAMPLE = "Request a new FIT sample now"
REVIEWER = "Synthetic FIT reviewer"


def reviewed_fit_case():
    case, item, model = fit_case()
    iid = item["id"]
    case["author"] = REVIEWER
    case = core.normalize_import(case)
    case = workflow.save_model(case, iid, model, REVIEWER)
    case = workflow.accept_scenario(case, iid, "fit-positive", REVIEWER, "Synthetic report review.")
    options = [
        option("c-positive", "colonoscopy", COLONOSCOPY, True),
        option("c-negative", "annual", ANNUAL, rule_id="fit-negative-rule") | {
            "attractive_fact_ids": ["fact-lab_status"], "appropriate_in_scenario_ids": ["fit-negative"],
            "clinical_rationale": "The modeled annual interval requires a processed negative report.",
        },
        option("c-rejected", "repeat", NEW_SAMPLE, rule_id="fit-rejected-rule") | {
            "appropriate_in_scenario_ids": ["fit-rejected"],
            "clinical_rationale": "The modeled new sample requires a rejected specimen.",
        },
    ]
    case = workflow.assemble_candidate_options(case, iid, {
        "scenario_id": "fit-positive", "options": options, "rejected_candidates": [], "shortage_count": 1,
    }, REVIEWER)
    return core.normalize_import(case), iid


def table(app, column):
    tables = [element.value for element in app.dataframe if column in element.value.columns]
    assert len(tables) == 1, (column, len(tables))
    return tables[0].set_index(column)


def rendered(app):
    assert not app.exception, [element.value for element in app.exception]
    return "\n".join(element.value for element in app.markdown)


@pytest.mark.parametrize("entry", [
    APP / "app.py", APP / "cloud_entry.py", APP.parent / "deploy/item_studio/streamlit_app.py",
], ids=["local", "hosted", "deployment-wrapper"])
def test_fit_feature_and_distractor_views(entry, tmp_path, monkeypatch):
    monkeypatch.setenv("ITEMDEV_HOSTED", "0")
    monkeypatch.setenv("ITEMDEV_DATA_DIR", str(tmp_path))

    def forbid_http(*args, **kwargs):
        pytest.fail("The manual FIT authoring exercise made an external request.")

    monkeypatch.setattr(ai_adapter, "_http", forbid_http)
    case, iid = reviewed_fit_case()
    cid = case["id"]
    hosted = entry.name != "app.py"
    app = AppTest.from_file(str(entry), default_timeout=30)
    if hosted:
        app.secrets["test_fixture"] = "synthetic"
        app.session_state["itemdev_hosted_library"] = {
            "schema_version": 1, "cases": {cid: case}, "history": {}, "redo": {}, "events": {},
        }
    else:
        storage.save_case(case)

    def saved():
        return app.session_state["itemdev_cases"][cid]

    def selected():
        return workflow.item_for(saved(), iid)["decision_workflow"]["selection"]

    work = f"{cid}::decision:{iid}:"
    spaces = work + "spaces:fit-positive:"
    app.run()
    assert not app.toggle(key="itemdev_ai_enabled").value
    assert f"**Evaluated key** · {COLONOSCOPY}" in rendered(app)
    assert {"Feature space", "Distractor space"} <= {t.label for t in app.tabs}
    expected = table(app, "Feature")
    result = expected.loc["fit_result"]
    assert result["Value"] == "positive"
    assert result["Impact"] == "decisive"
    assert result["Variation"] == "variable"
    assert result["Coupled features"] == "lab_status"
    assert result["Feasibility constraints"] == "final-report"
    feature_tab = next(t for t in app.tabs if t.label == "Feature space")
    feasible = table(feature_tab, "Scenario")
    assert feasible["Key"].to_dict() == {
        "fit-positive": COLONOSCOPY, "fit-negative": ANNUAL, "fit-rejected": NEW_SAMPLE,
    }

    candidates = table(app, "Candidate")
    assert candidates.loc[COLONOSCOPY, "Current eligibility"] == "keyed"
    for text, sid in [(ANNUAL, "fit-negative"), (NEW_SAMPLE, "fit-rejected")]:
        assert candidates.loc[text, "Decision"] == "Selected"
        assert candidates.loc[text, "Where keyed"] == sid
        assert candidates.loc[text, "Excluding rules"] == sid + "-rule"
        assert candidates.loc[text, "Clinical judgment"]
    assert "Shortfall **1**" in rendered(app)

    # Record the SME's explicit judgment in the normal candidate editor.
    app.selectbox(key=spaces + "candidate").set_value("c-negative").run()
    candidate_form = spaces + "edit_candidate:c-negative:"
    app.selectbox(key=candidate_form + "decision").set_value("Eligible")
    app.multiselect(key=candidate_form + "bank").set_value(["bank-1"])
    app.button(key=f"FormSubmitter:{candidate_form}-Save candidate decision").click().run()
    rendered(app)
    assert table(app, "Candidate").loc[ANNUAL, "Decision"] == "Eligible"
    app.selectbox(key=spaces + "candidate").set_value("c-negative").run()
    assert any(e.label == "Inspect source · bank-1" for e in app.expander)
    assert any("Synthetic bank summary" in element.value for element in app.text)

    # An impossible report cannot be applied. A valid neighboring report is a preview until confirmed.
    version = saved()["version"]
    trial_report = spaces + f"trial:fit_result:{version}"
    trial_status = spaces + f"trial:lab_status:{version}"
    app.selectbox(key=trial_status).set_value(1).run()
    assert "**Trial outcome** · infeasible" in rendered(app)
    assert app.button(key=spaces + "confirm_trial").disabled
    app.selectbox(key=trial_status).set_value(0)
    app.selectbox(key=trial_report).set_value(1).run()
    assert f"Trial key · {ANNUAL}" in rendered(app)
    assert f"**Evaluated key** · {ANNUAL}" in rendered(app)
    assert saved()["version"] == version
    assert selected()["scenario_id"] == "fit-positive"
    assert next(f["value"] for f in saved()["facts"] if f["id"] == "fact-fit_result") == "positive"
    assert table(app, "Candidate").loc[ANNUAL, "Decision"] == "Reassess"
    assert app.button(key=spaces + "assemble").disabled

    app.text_area(key=work + "notes").set_value("Reviewing the modeled negative result in a synthetic case.")
    app.button(key=spaces + "confirm_trial").click().run()
    assert f"**Evaluated key** · {ANNUAL}" in rendered(app)
    assert selected()["keyed_action_ids"] == ["annual"]
    assert selected()["scenario_id"] == app.selectbox(key=work + "scenario").value == "fit-negative"
    assert next(f["value"] for f in saved()["facts"] if f["id"] == "fact-fit_result") == "negative"
    rechecked = table(app, "Candidate")
    assert rechecked.loc[ANNUAL, "Decision"] == "Reassess"
    assert rechecked.loc[COLONOSCOPY, "Where keyed"] == "fit-positive"
    assert rechecked.loc[COLONOSCOPY, "Excluding rules"] == "fit-positive-rule"
    assert rechecked.loc[NEW_SAMPLE, "Where keyed"] == "fit-rejected"
    assert rechecked.loc[NEW_SAMPLE, "Excluding rules"] == "fit-rejected-rule"
    assert rechecked.loc[COLONOSCOPY, "Current eligibility"] == "review"

    app.button(key="undo:unfolding").click().run()
    assert f"**Evaluated key** · {COLONOSCOPY}" in rendered(app)
    assert selected()["keyed_action_ids"] == ["colonoscopy"]
    # Undo preserves the authored decision and dates its approval to the earlier revision.
    prior = workflow.item_for(saved(), iid)["decision_workflow"]["candidate_reviews"]["c-negative"]
    assert prior["decision"] == "Eligible" and prior["actor"] == REVIEWER
    assert table(app, "Candidate").loc[ANNUAL, "Decision"] == "Reassess"
    app.button(key="redo:unfolding").click().run()
    assert f"**Evaluated key** · {ANNUAL}" in rendered(app)
    assert selected()["keyed_action_ids"] == ["annual"]
    assert (tmp_path / "library.json").exists() is not hosted
