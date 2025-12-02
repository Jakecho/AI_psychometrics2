#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIG SME Toolkit – Streamlit GUI
Deconstruct & Reconstruct + Key Feature Venn Diagram

Features:
- Streamlit interface for SMEs to create high-order items
- OpenAI GPT-4o integration
- Interactive editing of stem, question, options, and rationales
- LLM-based validation and feedback
- Download finalized items as JSON

Requirements:
    pip install streamlit openai
"""

import streamlit as st
from openai import OpenAI
import json
import uuid
import os
from datetime import datetime
from typing import List, Dict, Any, Optional

# ============================================================
# Configuration
# ============================================================

def get_openai_client(api_key: str) -> OpenAI:
    """Initialize OpenAI client"""
    return OpenAI(api_key=api_key)

# ============================================================
# LLM Functions
# ============================================================

def generate_high_order_item(
    client: OpenAI,
    domain: str,
    subtopic: str,
    cognitive_level: str,
    seed_scenario: Optional[str],
    model: str = "gpt-4o",
    temperature: float = 0.4
) -> Dict[str, Any]:
    """Generate item using Deconstruct & Reconstruct + Key Feature Venn Diagram"""
    
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

Steps:
1) DECONSTRUCT:
   - Identify key cognitive operations required (analysis, prioritization, etc.).
   - Identify key domain concepts and their relationships.

2) RECONSTRUCT:
   - Write a concise clinical scenario (stem) that requires the target cognitive level.
   - Write a single best-answer question.

3) KEY FEATURE VENN DIAGRAM FOR OPTIONS:
   - Propose ONE correct option.
   - Propose at least 3-4 plausible distractors that share some—but not all—key features.
   - For each distractor, explain briefly why it is plausible and why it is incorrect.

Output strictly as JSON with this schema:
{{
  "stem": "...",
  "question": "...",
  "options": ["...", "...", "...", "..."],
  "correct_index": 0,
  "distractor_rationales": [
    "",
    "rationale for option 2",
    "rationale for option 3",
    "rationale for option 4"
  ],
  "deconstruction": {{
    "cognitive_level": "{cognitive_level}",
    "cognitive_steps": ["...", "..."],
    "content_features": ["...", "..."]
  }}
}}
"""
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return result
    except Exception as e:
        st.error(f"Error generating item: {e}")
        return None

def evaluate_item(
    client: OpenAI,
    item_data: Dict[str, Any],
    domain: str,
    subtopic: str,
    cognitive_level: str,
    model: str = "gpt-4o",
    temperature: float = 0.3
) -> str:
    """LLM-based evaluation of item quality"""
    
    prompt = f"""
You are a psychometric item quality reviewer.

Evaluate the following multiple-choice item.

Item:
STEM: {item_data['stem']}

QUESTION: {item_data['question']}

OPTIONS:
{json.dumps(item_data['options'], indent=2)}
Correct index: {item_data['correct_index']}

Domain: {domain}
Subtopic: {subtopic}
Target cognitive level: {cognitive_level}

Provide structured feedback with sections:
1) Cognitive complexity - Does it match the target level?
2) Domain/content accuracy - Is the content correct and current?
3) Logical coherence - Do stem, question, and options align?
4) Style/grammar/bias - Any issues with clarity or bias?
5) Distractor quality - Are distractors plausible and distinct?
6) Overall recommendation - Accept / Revise / Reject with brief justification.

Provide clear, actionable feedback.
"""
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Evaluation error: {e}"

# ============================================================
# Streamlit App
# ============================================================

def main():
    st.set_page_config(
        page_title="AIG SME Toolkit",
        page_icon="🧠",
        layout="wide"
    )
    
    st.title("🧠 AIG SME Toolkit")
    st.markdown("**High-Order Item Generation with Deconstruct & Reconstruct + Key Feature Venn Diagram**")
    
    # Sidebar - Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=os.getenv("OPENAI_API_KEY", ""),
            help="Enter your OpenAI API key"
        )
        
        if not api_key:
            st.warning("⚠️ Please enter your OpenAI API key to continue")
            st.stop()
        
        st.divider()
        
        # Model settings
        st.subheader("Model Settings")
        model = st.selectbox(
            "Model",
            ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
            index=0
        )
        
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.4,
            step=0.1,
            help="Higher = more creative, Lower = more focused"
        )
    
    # Initialize session state
    if 'item_data' not in st.session_state:
        st.session_state.item_data = None
    if 'evaluation' not in st.session_state:
        st.session_state.evaluation = None
    if 'finalized_items' not in st.session_state:
        st.session_state.finalized_items = []
    
    # Main content - Item Specification
    st.header("📋 Step 1: Item Specification")
    
    col1, col2 = st.columns(2)
    
    with col1:
        domain = st.text_input(
            "Domain",
            value="Internal Medicine",
            help="e.g., Internal Medicine, Pediatrics, Surgery"
        )
        
        cognitive_level = st.selectbox(
            "Target Cognitive Level",
            ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"],
            index=3,
            help="Bloom's taxonomy level"
        )
    
    with col2:
        subtopic = st.text_input(
            "Subtopic",
            value="Acute Coronary Syndrome",
            help="Specific clinical topic or condition"
        )
        
        seed_scenario = st.text_area(
            "Seed Scenario (Optional)",
            placeholder="Enter a seed scenario or leave blank for LLM to generate...",
            height=100,
            help="Optional starting scenario for the item"
        )
    
    # Generate button
    if st.button("🚀 Generate Item", type="primary", use_container_width=True):
        with st.spinner("Generating high-order item..."):
            client = get_openai_client(api_key)
            result = generate_high_order_item(
                client=client,
                domain=domain,
                subtopic=subtopic,
                cognitive_level=cognitive_level,
                seed_scenario=seed_scenario if seed_scenario.strip() else None,
                model=model,
                temperature=temperature
            )
            
            if result:
                st.session_state.item_data = result
                st.session_state.evaluation = None
                st.success("✅ Item generated successfully!")
    
    # Display and edit generated item
    if st.session_state.item_data:
        st.divider()
        st.header("✏️ Step 2: Edit Item")
        
        item = st.session_state.item_data
        
        # Stem
        st.subheader("Clinical Scenario (Stem)")
        edited_stem = st.text_area(
            "Stem",
            value=item['stem'],
            height=150,
            key="stem_edit"
        )
        
        # Question
        st.subheader("Question")
        edited_question = st.text_area(
            "Question",
            value=item['question'],
            height=80,
            key="question_edit"
        )
        
        # Options
        st.subheader("Options & Rationales")
        
        edited_options = []
        edited_rationales = []
        
        for i, (opt, rat) in enumerate(zip(item['options'], item['distractor_rationales'])):
            col1, col2 = st.columns([2, 3])
            
            with col1:
                is_key = i == item['correct_index']
                marker = " ✅ (Correct)" if is_key else ""
                edited_opt = st.text_area(
                    f"Option {i+1}{marker}",
                    value=opt,
                    height=80,
                    key=f"opt_{i}"
                )
                edited_options.append(edited_opt)
            
            with col2:
                edited_rat = st.text_area(
                    f"Rationale {i+1}",
                    value=rat if rat else "(Correct answer - no rationale needed)",
                    height=80,
                    key=f"rat_{i}",
                    disabled=is_key
                )
                edited_rationales.append(edited_rat if not is_key else "")
        
        # Correct answer selection
        correct_index = st.radio(
            "Select Correct Answer",
            options=list(range(len(edited_options))),
            index=item['correct_index'],
            format_func=lambda x: f"Option {x+1}",
            horizontal=True
        )
        
        # Add new distractor
        st.subheader("Add New Distractor")
        col1, col2 = st.columns([2, 3])
        with col1:
            new_opt = st.text_input("New Option Text", key="new_opt")
        with col2:
            new_rat = st.text_input("Rationale for New Option", key="new_rat")
        
        if st.button("➕ Add Distractor"):
            if new_opt.strip():
                edited_options.append(new_opt)
                edited_rationales.append(new_rat)
                st.success("Distractor added! Click 'Update Item' to save.")
        
        # Update item data
        if st.button("💾 Update Item", type="primary"):
            st.session_state.item_data.update({
                'stem': edited_stem,
                'question': edited_question,
                'options': edited_options,
                'correct_index': correct_index,
                'distractor_rationales': edited_rationales
            })
            st.success("✅ Item updated!")
        
        # Validation
        st.divider()
        st.header("🔍 Step 3: Validate Item")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔍 Evaluate with LLM", use_container_width=True):
                with st.spinner("Evaluating item quality..."):
                    client = get_openai_client(api_key)
                    evaluation = evaluate_item(
                        client=client,
                        item_data=st.session_state.item_data,
                        domain=domain,
                        subtopic=subtopic,
                        cognitive_level=cognitive_level,
                        model=model,
                        temperature=0.3
                    )
                    st.session_state.evaluation = evaluation
        
        with col2:
            if st.button("✅ Finalize Item", type="primary", use_container_width=True):
                finalized = {
                    'item_id': str(uuid.uuid4()),
                    'domain': domain,
                    'subtopic': subtopic,
                    'cognitive_level': cognitive_level,
                    'stem': st.session_state.item_data['stem'],
                    'question': st.session_state.item_data['question'],
                    'options': st.session_state.item_data['options'],
                    'correct_index': st.session_state.item_data['correct_index'],
                    'distractor_rationales': st.session_state.item_data['distractor_rationales'],
                    'deconstruction': st.session_state.item_data.get('deconstruction', {}),
                    'evaluation_summary': st.session_state.evaluation or "No evaluation performed",
                    'timestamp': datetime.now().isoformat(),
                    'created_by': 'SME'
                }
                st.session_state.finalized_items.append(finalized)
                st.success(f"✅ Item finalized! Total items: {len(st.session_state.finalized_items)}")
        
        # Display evaluation
        if st.session_state.evaluation:
            st.subheader("📊 Evaluation Results")
            st.info(st.session_state.evaluation)
        
        # Deconstruction info
        if 'deconstruction' in item:
            with st.expander("🧩 View Deconstruction Details"):
                st.json(item['deconstruction'])
    
    # Finalized items section
    if st.session_state.finalized_items:
        st.divider()
        st.header("📦 Step 4: Download Finalized Items")
        
        st.write(f"**Total Finalized Items:** {len(st.session_state.finalized_items)}")
        
        # Show finalized items
        for i, item in enumerate(st.session_state.finalized_items):
            with st.expander(f"Item {i+1}: {item['subtopic']} ({item['cognitive_level']})"):
                st.write(f"**ID:** {item['item_id']}")
                st.write(f"**Stem:** {item['stem'][:100]}...")
                st.write(f"**Question:** {item['question']}")
                st.write(f"**Correct Answer:** Option {item['correct_index'] + 1}")
        
        # Download buttons
        col1, col2 = st.columns(2)
        
        with col1:
            # Download as JSON
            json_data = json.dumps(st.session_state.finalized_items, indent=2, ensure_ascii=False)
            st.download_button(
                label="📥 Download as JSON",
                data=json_data,
                file_name=f"finalized_items_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        with col2:
            # Download as pretty text
            text_output = []
            for i, item in enumerate(st.session_state.finalized_items):
                text_output.append(f"=" * 80)
                text_output.append(f"ITEM {i+1}")
                text_output.append(f"=" * 80)
                text_output.append(f"ID: {item['item_id']}")
                text_output.append(f"Domain: {item['domain']}")
                text_output.append(f"Subtopic: {item['subtopic']}")
                text_output.append(f"Cognitive Level: {item['cognitive_level']}")
                text_output.append(f"\nSTEM:\n{item['stem']}")
                text_output.append(f"\nQUESTION:\n{item['question']}")
                text_output.append(f"\nOPTIONS:")
                for j, opt in enumerate(item['options']):
                    marker = " (CORRECT)" if j == item['correct_index'] else ""
                    text_output.append(f"{j+1}. {opt}{marker}")
                text_output.append(f"\nRATIONALES:")
                for j, rat in enumerate(item['distractor_rationales']):
                    if rat:
                        text_output.append(f"Option {j+1}: {rat}")
                text_output.append(f"\nEVALUATION:\n{item['evaluation_summary']}")
                text_output.append("\n")
            
            text_data = "\n".join(text_output)
            st.download_button(
                label="📄 Download as Text",
                data=text_data,
                file_name=f"finalized_items_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        # Clear all button
        if st.button("🗑️ Clear All Finalized Items", type="secondary"):
            st.session_state.finalized_items = []
            st.rerun()
    
    # Footer
    st.divider()
    st.markdown("""
    ---
    **AIG SME Toolkit** | Powered by OpenAI GPT-4o  
    Using Deconstruct & Reconstruct Method + Key Feature Venn Diagram Method
    """)

if __name__ == "__main__":
    main()
