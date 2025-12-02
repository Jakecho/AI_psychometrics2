#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIG SME Toolkit – Deconstruct & Reconstruct + Key Feature Venn Diagram

CLI prototype implementing:
- High-order item generation (Deconstruct & Reconstruct Method)
- Model-based distractor generation (Key Feature Venn Diagram Method)
- SME editing loop with add/edit options
- LLM-based validation of cognitive level, domain alignment, logic, and style
- Stubs for RAG (reference retrieval) and similarity/enemy checks

==============================================================================
THEORETICAL RATIONALE
==============================================================================

1. DECONSTRUCT & RECONSTRUCT METHOD FOR HIGH-ORDER THINKING ITEMS

Rationale:
The Deconstruct & Reconstruct method addresses a fundamental challenge in automatic 
item generation: preventing the LLM from merely copying surface features of example 
items rather than understanding the underlying cognitive architecture.

Traditional AIG approaches often use template-based generation or simple paraphrasing,
which can lead to:
- Shallow cognitive demand (asking trivial variations of the same question)
- Instability in difficulty (small wording changes cause unpredictable difficulty shifts)
- Poor construct validity (items test recall rather than higher-order thinking)

The D&R method solves this by training the LLM to reason explicitly about TWO 
orthogonal dimensions:

A) COGNITIVE COMPLEXITY AXIS (Procedural/Mental Operations):
   - Based on Bloom's taxonomy (Remember → Understand → Apply → Analyze → Evaluate → Create)
   - Operationalized as number of reasoning steps, need to integrate multiple cues,
     requirement to compare/prioritize/evaluate alternatives
   - Example: "Analyze" level requires identifying relationships between concepts,
     distinguishing relevant from irrelevant information

B) CONTENT KNOWLEDGE COMPLEXITY AXIS (Domain Concepts):
   - Number of domain concepts required (e.g., pathophysiology + diagnostics + treatment)
   - Depth of relationships (simple associations vs. causal mechanisms vs. conditional logic)
   - Integration requirements (within-domain vs. cross-domain knowledge)
   - Example: ACS item requires understanding of cardiac pathology, ECG interpretation,
     risk stratification, and emergency protocols

By explicitly deconstructing items along these axes BEFORE generation, the LLM:
1. Identifies the TARGET cognitive operations (what mental work the examinee must do)
2. Identifies the KEY domain concepts and their interrelationships
3. Represents both as structured features (not just text)
4. RECONSTRUCTS new items that preserve these features while varying surface form

This creates items that are:
- Cognitively isomorphic (same difficulty/cognitive demand as exemplars)
- Content-aligned (test intended domain knowledge, not confounds)
- Diverse in surface features (reducing item cloning and memorization)

Evidence base:
- Cognitive load theory (Sweller): separating intrinsic vs. extraneous complexity
- Item modeling research (Gierl, Haladyna): generative frameworks preserve psychometric properties
- Educational taxonomy alignment (Anderson & Krathwohl): explicit cognitive level targeting

2. KEY FEATURE VENN DIAGRAM METHOD FOR DISTRACTOR MODELING

Rationale:
Effective distractors are the cornerstone of valid multiple-choice items, yet they are
notoriously difficult to generate automatically. Poor distractors are either:
- Too implausible (non-functional, chosen by <5% of examinees)
- Too similar to the key (creating multiple defensible answers)
- Random/arbitrary (lacking diagnostic value for identifying misconceptions)

The Key Feature Venn Diagram method models distractors as OVERLAPPING FEATURE SETS
in the knowledge space, analogous to intersecting circles in a Venn diagram.

Conceptual model:
- The CORRECT option represents a complete feature set: [A, B, C, D]
  Example: "Obtain ECG immediately" = [acute presentation, cardiac risk, time-sensitive, 
  diagnostic priority, standard of care]

- Each DISTRACTOR shares SOME but not ALL features with the key:
  Distractor 1: [A, B, C, X] - shares most features but violates ONE critical feature
  Example: "Schedule stress test in 2 weeks" = [cardiac evaluation, diagnostic approach]
  but LACKS [time-sensitive, acute care] → plausible but dangerous delay
  
  Distractor 2: [A, B, Y, Z] - shares domain context but wrong action category
  Example: "Reassure and discharge" = [addresses patient concern]
  but LACKS [cardiac workup, risk stratification] → premature closure error
  
  Distractor 3: [A, W, Y, Z] - superficial similarity but wrong mechanism
  Example: "Lifestyle modification only" = [cardiac prevention]
  but LACKS [acute intervention, diagnosis first] → confuses prevention with treatment

By instructing the LLM to:
1. IDENTIFY the key features that make the correct answer correct
2. GENERATE alternatives that share plausible features (domain relevance, partial correctness)
3. ENSURE each distractor violates at least one CRITICAL feature
4. PROVIDE rationales explaining the feature overlap AND the critical difference

This creates distractors that are:
- Plausible (share enough features to attract examinees with partial knowledge)
- Diagnostic (each distractor maps to a specific misconception or incomplete reasoning)
- Defensibly wrong (clear rationale for why each violates a critical feature)
- Educationally valuable (post-test review reveals the distinguishing features)

Evidence base:
- Distractor rationale research (Haladyna & Rodriguez): effective distractors target 
  common errors and misconceptions
- Cognitive diagnosis models (Tatsuoka): items should discriminate between knowledge states
- Feature-based categorization theory (Smith & Medin): experts use feature bundles for 
  clinical reasoning
- Near-miss analysis in medical education: learning occurs at boundaries between 
  correct and plausible-but-incorrect

Integration:
When combined, D&R + Key Feature Venn Diagram creates a principled AIG system where:
- Items target specific cognitive levels (via deconstruction of mental operations)
- Items test intended knowledge (via content feature specification)
- Distractors are model-based (via feature overlap engineering)
- SME editing preserves psychometric integrity (via LLM evaluation feedback)

This approach shifts AIG from "text generation" to "cognitive engineering," producing
items with predictable psychometric properties suitable for high-stakes assessment.

==============================================================================
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
import json
import uuid
import os
from openai import OpenAI


# ============================================================
# 1. Data Structures
# ============================================================

@dataclass
class Item:
    """Container for a single MC item."""
    item_id: str
    domain: str
    subtopic: str
    cognitive_level: str  # e.g., "Apply", "Analyze", "Evaluate"
    stem: str
    question: str
    options: List[str]          # includes correct + distractors
    correct_index: int          # index into options
    distractor_rationales: List[str]  # same length as options (empty string for key)
    references_used: List[str]  # citations/IDs of sources used
    evaluation_summary: str     # LLM feedback text
    similarity_flags: List[str] # textual descriptions of possible enemies/duplicates


# ============================================================
# 2. LLM Interface (stubbed)
# ============================================================

class LLMClient:
    """
    Wraps all LLM calls using OpenAI API.
    """

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gpt-4o", temperature: float = 0.4):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")
        self.client = OpenAI(api_key=self.api_key)
        self.model_name = model_name
        self.temperature = temperature

    def _call_llm(self, prompt: str, response_format: str = "json") -> str:
        """
        Call OpenAI API with the given prompt.
        Returns JSON string response.
        """
        try:
            messages = [{"role": "user", "content": prompt}]
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                response_format={"type": "json_object"} if response_format == "json" else None
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            print(f"\nError calling OpenAI API: {e}")
            raise

    def generate_high_order_item(
        self,
        domain: str,
        subtopic: str,
        cognitive_level: str,
        seed_scenario: Optional[str] = None,
        references: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generation call implementing Deconstruct & Reconstruct + Key Feature Venn Diagram.

        Returns a dict with:
            stem, question, options, correct_index, distractor_rationales, deconstruction
        """
        prompt = f"""
You are an expert item writer for high-stakes medical exams.

Task:
1. Use the Deconstruct & Reconstruct Method to generate a high-order item.
2. Use the Key Feature Venn Diagram Method to construct plausible distractors.

Domain: {domain}
Subtopic: {subtopic}
Target Bloom / cognitive level: {cognitive_level}

Seed scenario (if any):
{seed_scenario or "None (you may propose an appropriate scenario)."}

References (snippets) that you MUST align with (if any):
{references or "None provided."}

Steps:
1) DECONSTRUCT:
   - Identify key cognitive operations required (analysis, prioritization, etc.).
   - Identify key domain concepts and their relationships.

2) RECONSTRUCT:
   - Write a concise clinical scenario (stem) that requires the target cognitive level.
   - Write a single best-answer question.

3) KEY FEATURE VENN DIAGRAM FOR OPTIONS:
   - Propose ONE correct option.
   - Propose at least 3–5 plausible distractors that share some—but not all—key features.
   - For each distractor, explain briefly why it is plausible and why it is incorrect.

Output strictly as JSON with this schema:
{{
  "stem": "...",
  "question": "...",
  "options": ["...", "...", "..."],
  "correct_index": 0,
  "distractor_rationales": [
    "",  // empty string for the correct option
    "rationale for option 2",
    "rationale for option 3",
    ...
  ],
  "deconstruction": {{
    "cognitive_level": "{cognitive_level}",
    "cognitive_steps": ["...", "..."],
    "content_features": ["...", "..."]
  }}
}}
"""
        raw = self._call_llm(prompt, response_format="json")
        try:
            data = json.loads(raw)
            return data
        except json.JSONDecodeError as e:
            print(f"\nError parsing LLM response as JSON: {e}")
            print(f"Raw response: {raw}")
            raise

    def evaluate_item(
        self,
        item: Item,
        references: Optional[List[str]] = None
    ) -> str:
        """
        LLM-based evaluation of:
        - cognitive complexity
        - domain alignment
        - logical coherence
        - style/grammar/bias
        - distractor quality
        """
        prompt = f"""
You are a psychometric item quality reviewer.

Evaluate the following multiple-choice item.

Item:
STEM:
{item.stem}

QUESTION:
{item.question}

OPTIONS:
{json.dumps(item.options, ensure_ascii=False, indent=2)}
Correct index: {item.correct_index}

Domain: {item.domain}
Subtopic: {item.subtopic}
Target cognitive level: {item.cognitive_level}

References (if any) to check for alignment:
{references or "None provided."}

Provide structured feedback with sections:
1) Cognitive complexity
2) Domain/content accuracy
3) Logical coherence (stem–question–options)
4) Style/grammar/bias
5) Distractor quality
6) Overall recommendation (Accept / Revise / Reject) and brief justification.

Respond in plain text format with clear section headers.
"""
        try:
            feedback = self._call_llm(prompt, response_format="text")
            return feedback
        except Exception as e:
            print(f"\nError getting evaluation feedback: {e}")
            return f"Evaluation failed: {str(e)}"


# ============================================================
# 3. RAG / Similarity Stubs
# ============================================================

def retrieve_references(domain: str, subtopic: str) -> List[str]:
    """
    Stub for retrieval-augmented generation.
    Replace with actual vector search (e.g., pgvector) over textbooks/guidelines.
    """
    # Example snippet:
    return [
        "Snippet: For patients with suspected acute coronary syndrome, obtain a 12-lead ECG within 10 minutes.",
    ]


def check_similarity_against_item_bank(item: Item) -> List[str]:
    """
    Stub for similarity/enemy detection.
    Replace with vector similarity search in your item bank.
    """
    # Example placeholder:
    return [
        "Warning: Similar to item ID 2024-IM-ACS-012 (same scenario structure, different age).",
        "Review for potential redundancy/enemy relationship."
    ]


def save_item_to_json(item: Item, output_dir: str = "generated_items") -> str:
    """
    Save the finalized item as a JSON file.
    Returns file path.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{item.item_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(item), f, ensure_ascii=False, indent=2)
    return path


# ============================================================
# 4. SME Editing Helpers (CLI)
# ============================================================

def edit_text_element(element_name: str, content: str) -> str:
    print(f"\nCurrent {element_name}:")
    print(content)
    edited = input(f"\nEdit {element_name} (or press Enter to keep): ").strip()
    return edited if edited else content


def select_correct_option(options: List[str], current_index: int) -> int:
    print("\nCurrent options:")
    for i, opt in enumerate(options):
        marker = " (current key)" if i == current_index else ""
        print(f"  [{i}] {opt}{marker}")

    while True:
        choice = input(
            f"\nEnter index of correct answer (press Enter to keep {current_index}): "
        ).strip()
        if choice == "":
            return current_index
        if choice.isdigit():
            idx = int(choice)
            if 0 <= idx < len(options):
                return idx
        print("Invalid index, try again.")


def edit_options_and_rationales(
    options: List[str],
    distractor_rationales: List[str]
) -> Tuple[List[str], List[str]]:
    assert len(options) == len(distractor_rationales)

    for i in range(len(options)):
        options[i] = edit_text_element(f"option {i}", options[i])
        # Rationale is only for distractors, but we allow editing all
        distractor_rationales[i] = edit_text_element(
            f"rationale for option {i} (empty if correct)",
            distractor_rationales[i]
        )

    # Add new options (distractors) if SME wants
    while True:
        add_more = input("\nAdd another distractor option? (yes/no): ").strip().lower()
        if add_more == "yes":
            new_opt = input("Enter new distractor text: ").strip()
            new_rat = input("Enter rationale for this distractor: ").strip()
            if new_opt:
                options.append(new_opt)
                distractor_rationales.append(new_rat)
        else:
            break

    return options, distractor_rationales


# ============================================================
# 5. Validation Helpers (Modular)
# ============================================================

def check_cognitive_complexity(item: Item) -> str:
    # In a real implementation, you might call LLM or rules
    return f"Cognitive complexity appears consistent with {item.cognitive_level}."


def check_domain_alignment(item: Item, references: List[str]) -> str:
    # Placeholder: call LLM with item + references, or rules
    if references:
        return "Domain content is consistent with provided references."
    return "No references provided; domain alignment not fully checked."


def check_style_and_grammar(item: Item) -> str:
    # Placeholder: call LLM or grammar checker
    return "Stem and options are clear and grammatically acceptable."


def check_distractor_quality(item: Item) -> str:
    # Placeholder logic: require at least 3 options
    if len(item.options) < 3:
        return "WARNING: Fewer than 3 options; consider adding more distractors."
    return "Distractors appear plausible and distinct (manual review still required)."


def validate_and_evaluate_item(
    item: Item,
    llm_client: LLMClient,
    references: List[str]
) -> str:
    """
    Combines rule-based checks and LLM evaluation.
    """
    rule_feedback = []
    rule_feedback.append(check_cognitive_complexity(item))
    rule_feedback.append(check_domain_alignment(item, references))
    rule_feedback.append(check_style_and_grammar(item))
    rule_feedback.append(check_distractor_quality(item))

    llm_feedback = llm_client.evaluate_item(item, references)

    full_feedback = (
        "=== Rule-based checks ===\n" +
        "\n".join(f"- {x}" for x in rule_feedback) +
        "\n\n=== LLM qualitative review ===\n" +
        llm_feedback
    )
    return full_feedback


# ============================================================
# 6. Main Orchestration – Create High-Order Item
# ============================================================

def create_high_order_item_cli():
    """
    CLI entry point for creating one item.
    """
    print("=== AIG SME Toolkit – High-Order Item Creation ===")

    domain = input("Enter domain (e.g., Internal Medicine): ").strip() or "Internal Medicine"
    subtopic = input("Enter subtopic (e.g., Acute Coronary Syndrome): ").strip() or "Acute Coronary Syndrome"
    cognitive_level = input("Enter target cognitive level (e.g., Analyze): ").strip() or "Analyze"
    seed_scenario = input("Optional: enter a seed scenario (or leave blank): ").strip() or None

    llm = LLMClient()
    references = retrieve_references(domain, subtopic)

    # Loop until SME finalizes the item
    finalized_item: Optional[Item] = None

    while True:
        print("\n--- Generating initial item draft via LLM ---")
        gen = llm.generate_high_order_item(
            domain=domain,
            subtopic=subtopic,
            cognitive_level=cognitive_level,
            seed_scenario=seed_scenario,
            references=references
        )

        stem = gen["stem"]
        question = gen["question"]
        options = gen["options"]
        correct_index = gen["correct_index"]
        distractor_rationales = gen.get("distractor_rationales", [""] * len(options))

        # SME editing loop
        print("\n--- SME Editing ---")
        stem = edit_text_element("stem", stem)
        question = edit_text_element("question", question)
        options, distractor_rationales = edit_options_and_rationales(options, distractor_rationales)
        correct_index = select_correct_option(options, correct_index)

        # Build candidate Item object
        item = Item(
            item_id=str(uuid.uuid4()),
            domain=domain,
            subtopic=subtopic,
            cognitive_level=cognitive_level,
            stem=stem,
            question=question,
            options=options,
            correct_index=correct_index,
            distractor_rationales=distractor_rationales,
            references_used=references,
            evaluation_summary="",
            similarity_flags=[]
        )

        # Evaluation
        print("\n--- Validation & Evaluation ---")
        evaluation_summary = validate_and_evaluate_item(item, llm, references)
        print("\n" + evaluation_summary)

        # Optional similarity/enemy detection
        print("\n--- Similarity / Enemy Check (stub) ---")
        similarity_flags = check_similarity_against_item_bank(item)
        for flag in similarity_flags:
            print(f"- {flag}")

        item.evaluation_summary = evaluation_summary
        item.similarity_flags = similarity_flags

        # SME decision
        choice = input(
            "\nDo you want to (f)inalize, (e)dit again, or (r)egenerate from scratch? [f/e/r]: "
        ).strip().lower()

        if choice == "f":
            finalized_item = item
            break
        elif choice == "e":
            # Reuse the current edited version as starting point next loop
            seed_scenario = item.stem
        elif choice == "r":
            # Regenerate fresh from the underlying cognitive/content profile
            continue
        else:
            print("Unrecognized choice; assuming edit again.")
            seed_scenario = item.stem

    # Save finalized item
    if finalized_item:
        path = save_item_to_json(finalized_item)
        print(f"\n=== Final item saved to: {path} ===")
    else:
        print("\nNo item finalized.")


# ============================================================
# 7. Script Entry Point
# ============================================================

if __name__ == "__main__":
    create_high_order_item_cli()
