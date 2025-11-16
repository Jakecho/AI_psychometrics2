#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit GUI – NCLEX Item Generator via OpenAI
-----------------------------------------------

Requirements:
    pip install streamlit openai

Environment:
    - Set your OpenAI API key, e.g.:
        export OPENAI_API_KEY="sk-xxxxx"
      or on Windows:
        setx OPENAI_API_KEY "sk-xxxxx"

This app calls OpenAI chat models instead of Ollama.
"""

import json
import os
import textwrap
from typing import Optional, Dict, Any

import streamlit as st
from openai import OpenAI

# ============================================================
# 0. OpenAI client (initialized dynamically in app)
# ============================================================

# ============================================================
# 1. SYSTEM PROMPT FOR NCLEX ITEM MODELING
# ============================================================

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert NCLEX item writer and nurse educator specializing in high-quality NCLEX-style items for entry-level registered nurses.

Context:
- Items are for the NCLEX-RN examination (or comparable entry-level RN licensure exam).
- Items must reflect current nursing standards of practice and patient safety principles.
- Items should focus on clinical reasoning, prioritization, delegation, and safe, effective nursing care.

General NCLEX rules:
- Target the specified exam context and examinee level (entry-level RN).
- Use clear, concise, and realistic nursing scenarios.
- Focus on patient safety, ABCs (airway, breathing, circulation), Maslow, and nursing process (ADPIE) where appropriate.
- Ensure exactly ONE BEST answer for standard single-best-answer multiple-choice items.
- Avoid "all of the above" / "none of the above" options.
- Avoid real patient identifiers or unnecessary demographic details.
- Avoid biased, stigmatizing, or culturally insensitive language.
- Do not contradict standard scope-of-practice boundaries (RN vs LPN vs UAP).
- Base content on widely accepted, contemporary nursing and medical guidelines.

Your tasks:
- Generate NCLEX-style items from a provided focus statement and blueprint constraints.
- Emphasize the specified NCLEX focus (e.g., priority, delegation, assessment, teaching).
- Provide brief rationales for correct and incorrect options.
- Provide metadata useful to psychometricians and SMEs (client needs category, cognitive level, difficulty, etc.).

Output rules:
- Follow the requested JSON schema EXACTLY.
- Return ONLY valid JSON (no extra comments, no Markdown, no explanations outside the JSON).
- If the user’s instructions conflict with patient safety or accepted practice, follow the safest reasonable standard and record a note in the "warnings" field in the JSON.
""")


# ============================================================
# 2. USER PROMPT BUILDER (NCLEX)
# ============================================================

def build_nclex_user_prompt(
    exam_name: str,
    target_level: str,
    clinical_setting: str,
    focus_statement: str,
    nclex_focus: str,
    client_needs_category: str,
    client_situation: str,
    cognitive_level: str,
    intended_difficulty: str,
    max_scenario_words: int = 120,
    response_format: str = "single_best_answer",  # or "select_all_that_apply" / "short_answer"
    num_options: int = 4,
    num_items: int = 1,
    pitfalls: Optional[str] = None,
    item_purpose: str = "pretest"
) -> str:
    """
    Build the user prompt text for NCLEX item modeling.
    """

    if pitfalls is None or pitfalls.strip() == "":
        pitfalls = "none"

    # Core instructions and blueprint
    core = f"""
TASK
You will generate NCLEX-style items for an entry-level RN licensure exam, following the schema and constraints below.

EXAM CONTEXT
- Exam name: {exam_name}
- Target examinee level: {target_level}
- Clinical setting: {clinical_setting}

NCLEX FOCUS
- Focus statement (1–2 sentences):
  {focus_statement}
- NCLEX focus:
  {nclex_focus}
- Common pitfalls or enemy options to consider (optional):
  {pitfalls}

BLUEPRINT CONSTRAINTS
- Client needs category: {client_needs_category}
- Client situation / brief clinical context: {client_situation}
- Cognitive level: {cognitive_level}
- Intended difficulty: {intended_difficulty}
- Maximum scenario length: {max_scenario_words} words
- Item response format: {response_format}
- Number of options per MCQ (if applicable): {num_options}

ITEM GENERATION REQUIREMENTS
- Number of distinct items to generate: {num_items}
- Each item should:
  - Be a realistic NCLEX-style question.
  - Reflect the specified NCLEX focus (e.g., priority, delegation, assessment, teaching).
  - Use plausible but clearly differentiable options.
- Avoid reusing identical wording across items except for necessary technical terms.

ADDITIONAL RULES
- Use realistic but concise clinical details.
- Avoid excessive lab lists; include only findings that matter for the decision.
- Ensure each standard MCQ has exactly one BEST answer.
- Distractors should be plausible and represent common errors or misconceptions, not obviously wrong options.
""".strip()

    # JSON schema / output format specification
    # (note: contains comments in the *prompt*, but the model is instructed to return clean JSON)
    json_schema = f"""
OUTPUT FORMAT
Return the result as VALID JSON ONLY, with NO additional text or comments.

Use this exact JSON structure:

{{
  "items": [
    {{
      "item_id": "NCLEX_<short_descriptor>_<index>",
      "scenario": {{
        "title": "Short scenario title (e.g., 'Postoperative client with hypotension')",
        "text": "NCLEX-style stem / scenario text (<= {max_scenario_words} words)."
      }},
      "questions": [
        {{
          "question_id": "Q1",
          "lead_in": "Exact NCLEX-style question (e.g., 'Which action should the nurse take first?').",
          "response_format": "{response_format}",
          "options": [
            {{
              "label": "A",
              "text": "Option text.",
              "is_key": true,
              "rationale": "Why this option is the best answer based on NCLEX principles (e.g., safety, priority, scope of practice)."
            }},
            {{
              "label": "B",
              "text": "Option text.",
              "is_key": false,
              "rationale": "Why this option is not the best choice (common error or misconception)."
            }}
          ],
          "short_answer_key": null
        }}
      ],
      "focus_statement": "Copy or paraphrase the NCLEX focus this item targets.",
      "nclex_focus": "{nclex_focus}",
      "metadata": {{
        "exam_name": "{exam_name}",
        "target_level": "{target_level}",
        "client_needs_category": "{client_needs_category}",
        "clinical_setting": "{clinical_setting}",
        "cognitive_level": "{cognitive_level}",
        "intended_difficulty": "{intended_difficulty}",
        "blueprint_tags": [
          "NCLEX",
          "{client_needs_category}",
          "{nclex_focus}"
        ],
        "item_purpose": "{item_purpose}",
        "notes_for_reviewers": "Short note for SMEs or psychometricians.",
        "warnings": "Describe any safety, guideline, or scope-of-practice concerns. Use empty string if none."
      }}
    }}
  ]
}}

CONSTRAINTS
- Do NOT include any text outside the JSON.
- The JSON must be syntactically valid (no trailing commas, matching quotes/brackets).
- For standard single_best_answer items, ensure exactly one option has "is_key": true.
- For select_all_that_apply, multiple options may have "is_key": true but at least one must be correct.
- Ensure content is safe, consistent with NCLEX-style expectations, and aligned with contemporary nursing standards.
""".strip()

    return core + "\n\n" + json_schema


# ============================================================
# 3. OPENAI CALL WRAPPER
# ============================================================

def call_openai_chat(
    system_prompt: str,
    user_prompt: str,
    model: str,
    api_key: str,
    temperature: float = 0.3,
    max_tokens: int = 2048,
) -> str:
    """
    Call OpenAI chat completions with a system + user prompt.
    Returns the raw assistant content string.
    """
    if not api_key or not api_key.strip():
        raise ValueError(
            "OpenAI API key is required. Please enter your API key in the sidebar."
        )

    # Create client with provided API key
    client = OpenAI(api_key=api_key)

    # Prepare completion parameters - newer models use max_completion_tokens
    completion_params = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
    }
    
    # GPT-5-mini only supports default temperature (1)
    if model not in ["gpt-5-mini"]:
        completion_params["temperature"] = temperature
    
    # Use max_completion_tokens for newer models, max_tokens for older models
    if model in ["gpt-5-mini", "gpt-4o", "gpt-4o-mini"]:
        completion_params["max_completion_tokens"] = max_tokens
    else:
        completion_params["max_tokens"] = max_tokens

    try:
        response = client.chat.completions.create(**completion_params)
    except Exception as e:
        if "authentication" in str(e).lower() or "api_key" in str(e).lower():
            raise ValueError(
                f"Invalid API key. Please check your OpenAI API key. Error: {e}"
            )
        elif "rate_limit" in str(e).lower():
            raise RuntimeError(
                f"Rate limit exceeded. Please try again later. Error: {e}"
            )
        else:
            raise RuntimeError(f"OpenAI API error: {e}")

    # Extract the assistant message content
    try:
        choice = response.choices[0]
        content = choice.message.content
        
        # Check for refusal or empty content
        if content is None or not content.strip():
            finish_reason = getattr(choice, 'finish_reason', 'unknown')
            refusal = getattr(choice.message, 'refusal', None)
            
            error_msg = f"Model '{model}' returned empty content.\n"
            error_msg += f"Finish reason: {finish_reason}\n"
            
            if finish_reason == 'length':
                error_msg += "\nThe response was cut off due to token limits. Try:\n"
                error_msg += "• Reducing the number of items to generate\n"
                error_msg += "• Reducing max scenario length\n"
                error_msg += "• Using a different model (gpt-4o-mini or gpt-4o)\n"
            elif refusal:
                error_msg += f"Refusal reason: {refusal}\n"
            else:
                error_msg += "\nThis model may not be available or may have restrictions. Try using 'gpt-4o-mini' or 'gpt-4o' instead.\n"
            
            raise RuntimeError(error_msg)
            
    except (AttributeError, IndexError, KeyError) as e:
        raise RuntimeError(f"Unexpected OpenAI response format: {response}") from e

    return content


def generate_nclex_items(
    exam_name: str,
    target_level: str,
    clinical_setting: str,
    focus_statement: str,
    nclex_focus: str,
    client_needs_category: str,
    client_situation: str,
    cognitive_level: str,
    intended_difficulty: str,
    max_scenario_words: int = 120,
    response_format: str = "single_best_answer",
    num_options: int = 4,
    num_items: int = 1,
    pitfalls: Optional[str] = None,
    item_purpose: str = "pretest",
    model: str = "gpt-4o-mini",
    api_key: str = "",
    temperature: float = 0.3,
) -> Dict[str, Any]:
    """
    High-level wrapper for NCLEX items:
    - builds user prompt
    - calls OpenAI
    - parses JSON into Python dict
    """
    user_prompt = build_nclex_user_prompt(
        exam_name=exam_name,
        target_level=target_level,
        clinical_setting=clinical_setting,
        focus_statement=focus_statement,
        nclex_focus=nclex_focus,
        client_needs_category=client_needs_category,
        client_situation=client_situation,
        cognitive_level=cognitive_level,
        intended_difficulty=intended_difficulty,
        max_scenario_words=max_scenario_words,
        response_format=response_format,
        num_options=num_options,
        num_items=num_items,
        pitfalls=pitfalls,
        item_purpose=item_purpose,
    )

    raw = call_openai_chat(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=4096,
    )

    # Try to parse JSON
    try:
        if not raw or not raw.strip():
            raise ValueError(f"Model returned empty response. Model: {model}")
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        # If decoding fails, raise including raw text for debugging
        raise ValueError(f"Failed to parse JSON from model output.\nModel: {model}\nRaw output (first 500 chars):\n{raw[:500]}") from e

    return parsed


# ============================================================
# 4. STREAMLIT APP (GUI)
# ============================================================

def main():
    st.set_page_config(
        page_title="NCLEX Item Generator (OpenAI)",
        layout="wide",
    )

    # Password protection
    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False
    
    if not st.session_state['authenticated']:
        st.title("🔒 Access Protected")
        st.markdown("### NCLEX Item Generator (OpenAI)")
        st.info("This application is password protected. Please enter the password to continue.")
        
        password_input = st.text_input(
            "Enter Password:",
            type="password",
            key="password_input"
        )
        
        if st.button("🔓 Unlock", key="unlock_button"):
            if password_input == "AI_NCLEX_RN_v0":
                st.session_state['authenticated'] = True
                st.rerun()
            else:
                st.error("❌ Incorrect password. Access denied.")
        
        st.stop()
    
    st.title("🩺 NCLEX Item Generator (OpenAI)")

    # ---------------- Sidebar: Model & API settings ----------------
    st.sidebar.header("Model & API Settings")

    # Get default API key from secrets (for deployment) or environment variable (for local)
    default_api_key = ""
    try:
        # Try Streamlit secrets first (for Streamlit Cloud deployment)
        default_api_key = st.secrets.get("OPENAI_API_KEY", "")
    except (FileNotFoundError, AttributeError):
        # Fall back to environment variable (for local development)
        default_api_key = os.getenv("OPENAI_API_KEY", "")

    # API Key input - optional override
    api_key_input = st.sidebar.text_input(
        "OpenAI API Key (Optional)",
        type="password",
        value="",
        help="Enter your own OpenAI API key, or leave empty to use the default key.",
        placeholder="sk-proj-... (optional)"
    )
    
    # Use user's key if provided, otherwise fall back to default
    if api_key_input and api_key_input.strip():
        api_key = api_key_input
        # Show masked version
        if len(api_key_input) > 10:
            masked_key = api_key_input[:7] + "..." + "*" * 20 + "..." + api_key_input[-4:]
        else:
            masked_key = "*" * len(api_key_input)
        st.sidebar.caption(f"🔑 Using your key: `{masked_key}`")
    else:
        # Use default key
        api_key = default_api_key
        if api_key:
            st.sidebar.caption("🔑 Using default API key")

    # Model selection via dropdown (selectbox) – adjust to the models you have access to.
    model_options = [
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-5-mini"
    ]
    selected_model = st.sidebar.selectbox(
        "Select OpenAI model",
        options=model_options,
        index=0,
        help="GPT-4o-mini (recommended): Fast, reliable, cost-effective. GPT-4o: Higher quality. GPT-5-mini: Experimental.",
    )
    
    # Warning for GPT-5-mini model
    if selected_model == "gpt-5-mini":
        st.sidebar.warning(f"⚠️ {selected_model} is experimental and may not work reliably. Consider using gpt-4o-mini or gpt-4o for stable results.")

    # Advanced item generation settings
    st.sidebar.subheader("Item Generation Settings")

    response_format = st.sidebar.selectbox(
        "Response format",
        options=["single_best_answer", "select_all_that_apply", "short_answer"],
        index=0,
    )

    max_scenario_words = st.sidebar.slider(
        "Max scenario length (words)",
        min_value=60,
        max_value=200,
        value=120,
        step=10,
    )

    num_options = st.sidebar.slider(
        "Number of options per item (for MCQ/SATA)",
        min_value=4,
        max_value=6,
        value=4,
        step=1,
        help="Ignored if response_format = 'short_answer'.",
    )

    num_items = st.sidebar.slider(
        "Number of NCLEX items to generate",
        min_value=1,
        max_value=5,
        value=1,
        step=1,
    )

    item_purpose = st.sidebar.selectbox(
        "Item purpose",
        options=["pretest", "operational", "pilot"],
        index=0,
    )

    temperature = st.sidebar.slider(
        "Temperature (creativity)",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1,
        help="Lower = more focused, higher = more creative.",
    )

    # ---------------- Main layout: Inputs & Output ----------------
    col_left, col_right = st.columns([1.3, 1])

    with col_left:
        st.header("Input: NCLEX Focus & Blueprint")

        exam_name = st.text_input(
            "Exam name",
            value="NCLEX-RN",
        )

        target_level = st.text_input(
            "Target examinee level",
            value="Entry-level registered nurse",
        )

        clinical_setting = st.text_input(
            "Clinical setting",
            value="Acute care hospital",
        )

        client_needs_category = st.selectbox(
            "Client needs category",
            options=[
                "Safe and Effective Care Environment",
                "Health Promotion and Maintenance",
                "Psychosocial Integrity",
                "Physiological Integrity",
            ],
            index=0,
        )

        nclex_focus = st.selectbox(
            "NCLEX focus",
            options=[
                "priority/triage",
                "delegation/supervision",
                "assessment/data collection",
                "implementation/interventions",
                "teaching/learning",
                "pharmacology/safety",
            ],
            index=0,
        )

        cognitive_level = st.selectbox(
            "Cognitive level",
            options=["recall", "application", "analysis"],
            index=1,
        )

        intended_difficulty = st.selectbox(
            "Intended difficulty",
            options=["easy", "moderate", "hard"],
            index=1,
        )

        focus_statement = st.text_area(
            "NCLEX focus statement (1–2 sentences)",
            height=100,
            value=(
                "The nurse must determine which postoperative client to assess first, "
                "applying principles of priority (airway, breathing, circulation, and risk for deterioration)."
            ),
        )

        client_situation = st.text_area(
            "Client situation / brief context",
            height=100,
            value=(
                "Several adult postoperative clients on a surgical unit, each with different vital signs "
                "and symptoms that may indicate complications."
            ),
        )

        pitfalls = st.text_area(
            "Common pitfalls / enemy options (optional)",
            height=80,
            value="Choosing stable clients over unstable ones; ignoring subtle signs of airway or bleeding problems.",
        )

        generate_button = st.button("🚀 Generate NCLEX Items")

    with col_right:
        st.header("Output: Generated Items (JSON)")
        output_placeholder = st.empty()
        
        # Add custom CSS for color-coded output
        st.markdown("""
        <style>
        .scenario-title {
            color: #1f77b4;
            font-weight: bold;
            font-size: 18px;
            margin-top: 20px;
        }
        .question-text {
            color: #ff7f0e;
            font-weight: bold;
            font-size: 16px;
            margin-top: 15px;
        }
        .correct-answer {
            background-color: #d4edda;
            border-left: 4px solid #28a745;
            padding: 10px;
            margin: 5px 0;
        }
        .incorrect-answer {
            background-color: #f8d7da;
            border-left: 4px solid #dc3545;
            padding: 10px;
            margin: 5px 0;
        }
        .rationale {
            color: #6c757d;
            font-style: italic;
            font-size: 14px;
        }
        .metadata {
            background-color: #e7f3ff;
            padding: 10px;
            border-radius: 5px;
            margin-top: 10px;
        }
        </style>
        """, unsafe_allow_html=True)

    # ---------------- Handle Generation ----------------
    if generate_button:
        # Basic validation
        if not api_key or not api_key.strip():
            st.error("⚠️ OpenAI API Key is required. Please enter it in the sidebar.")
            return
            
        if not focus_statement.strip():
            st.error("NCLEX focus statement cannot be empty.")
            return

        with st.spinner("Generating NCLEX items via OpenAI..."):
            try:
                result = generate_nclex_items(
                    exam_name=exam_name,
                    target_level=target_level,
                    clinical_setting=clinical_setting,
                    focus_statement=focus_statement,
                    nclex_focus=nclex_focus,
                    client_needs_category=client_needs_category,
                    client_situation=client_situation,
                    cognitive_level=cognitive_level,
                    intended_difficulty=intended_difficulty,
                    max_scenario_words=max_scenario_words,
                    response_format=response_format,
                    num_options=num_options,
                    num_items=num_items,
                    pitfalls=pitfalls,
                    item_purpose=item_purpose,
                    model=selected_model,
                    api_key=api_key,
                    temperature=temperature,
                )

                # Store result in session state
                st.session_state['generated_result'] = result
                st.success("✅ NCLEX items generated successfully!")
                
            except Exception as exc:
                st.error(f"Error during generation: {exc}")
                # Clear any previous results on error
                if 'generated_result' in st.session_state:
                    del st.session_state['generated_result']
                return
                
    # Display output if result exists in session state
    if 'generated_result' in st.session_state:
        result = st.session_state['generated_result']
        items = result.get("items", [])
        
        with output_placeholder.container():
            # If multiple items, add dropdown selector
            if len(items) > 1:
                selected_item_idx = st.selectbox(
                    "Select Item to View:",
                    options=range(len(items)),
                    format_func=lambda i: f"Item {i+1}: {items[i].get('scenario', {}).get('title', 'N/A')}",
                    key="item_selector"
                )
                items_to_display = [items[selected_item_idx]]
                display_idx = selected_item_idx + 1
            else:
                items_to_display = items
                display_idx = 1
            
            # Display selected item(s)
            for item in items_to_display:
                scenario = item.get("scenario", {})
                st.markdown(f'<div class="scenario-title">📋 Item {display_idx}: {scenario.get("title", "N/A")}</div>', unsafe_allow_html=True)
                st.write(scenario.get("text", ""))
                
                for question in item.get("questions", []):
                    st.markdown(f'<div class="question-text">❓ {question.get("lead_in", "")}</div>', unsafe_allow_html=True)
                    
                    for option in question.get("options", []):
                        is_correct = option.get("is_key", False)
                        label = option.get("label", "")
                        text = option.get("text", "")
                        rationale = option.get("rationale", "")
                        
                        if is_correct:
                            st.markdown(f'''
                            <div class="correct-answer">
                                <strong>✓ {label}.</strong> {text}<br>
                                <div class="rationale">💡 {rationale}</div>
                            </div>
                            ''', unsafe_allow_html=True)
                        else:
                            st.markdown(f'''
                            <div class="incorrect-answer">
                                <strong>✗ {label}.</strong> {text}<br>
                                <div class="rationale">💡 {rationale}</div>
                            </div>
                            ''', unsafe_allow_html=True)
                
                # Metadata section
                metadata = item.get("metadata", {})
                st.markdown(f'''
                <div class="metadata">
                    <strong>📊 Metadata:</strong><br>
                    • Category: {metadata.get("client_needs_category", "N/A")}<br>
                    • Cognitive Level: {metadata.get("cognitive_level", "N/A")}<br>
                    • Difficulty: {metadata.get("intended_difficulty", "N/A")}<br>
                    • NCLEX Focus: {item.get("nclex_focus", "N/A")}
                </div>
                ''', unsafe_allow_html=True)
                
                if len(items) == 1:
                    st.markdown("---")
            
            # Also show raw JSON in expander
            with st.expander("📄 View Raw JSON"):
                st.json(result)

            # Optional: allow download
            st.download_button(
                label="💾 Download JSON",
                data=json.dumps(result, indent=2, ensure_ascii=False),
                file_name="nclex_items.json",
                mime="application/json",
            )


if __name__ == "__main__":
    main()
