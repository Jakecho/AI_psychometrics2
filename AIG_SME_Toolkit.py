#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIG SME Toolkit – Enhanced Streamlit GUI
Deconstruct & Reconstruct + Key Feature Venn Diagram

Features:
- Transparent LLM reasoning with expandable rationales
- Individual component regeneration (stem, question, each distractor)
- Source-aware validation (imported materials → LLM knowledge)
- 10 model-based distractors with feature overlap visualization
- Downloadable items with full reasoning traces

Requirements:
    pip install streamlit openai PyPDF2

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

import streamlit as st
from openai import OpenAI
import json
import uuid
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
import io

# Try to import PyPDF2 for PDF handling
try:
    import PyPDF2
    HAS_PDF = True
except ImportError:
    HAS_PDF = False
    st.warning("PyPDF2 not installed. PDF upload disabled. Install with: pip install PyPDF2")

# ============================================================
# LLM Functions with Reasoning Transparency
# ============================================================

def generate_stem_with_reasoning(
    client: OpenAI,
    domain: str,
    subtopic: str,
    cognitive_level: str,
    source_context: str = "",
    sample_template: str = "",
    model: str = "gpt-4o",
    temperature: float = 0.4
) -> Dict[str, Any]:
    """Generate clinical scenario with D&R reasoning"""
    
    source_instruction = f"\n\nIMPORTANT: Base your scenario on this source material:\n{source_context}" if source_context else ""
    
    template_instruction = f"\n\nSAMPLE ITEM TEMPLATE (analyze structure and cognitive approach):\n{sample_template}\n\nUse this template to understand the preferred item format, complexity level, and cognitive structure. Create a NEW item with similar characteristics but different content." if sample_template else ""
    
    prompt = f"""
You are an expert using the Deconstruct & Reconstruct Method to create high-order items.

Domain: {domain}
Subtopic: {subtopic}
Target Cognitive Level: {cognitive_level}{source_instruction}{template_instruction}

TASK:
1) DECONSTRUCT the cognitive requirements:
   - What mental operations must the examinee perform? (e.g., analyze relationships, prioritize actions, evaluate options)
   - How many reasoning steps are required?
   - What makes this {cognitive_level}-level thinking?

2) DECONSTRUCT the content requirements:
   - What domain concepts must be integrated?
   - What relationships between concepts are critical?
   - What clinical context best elicits this reasoning?

3) RECONSTRUCT a clinical scenario (stem):
   - Write a scenario that REQUIRES the cognitive operations you identified
   - Include cues that necessitate the content integration you specified
   - Keep it concise but rich enough to support the target cognitive level

Output as JSON:
{{
  "stem": "The clinical scenario text...",
  "deconstruct_reasoning": {{
    "cognitive_operations": ["operation 1", "operation 2", ...],
    "reasoning_steps": 3,
    "why_this_level": "Explanation of why this requires {cognitive_level}-level thinking",
    "content_concepts": ["concept 1", "concept 2", ...],
    "concept_relationships": "How concepts interact in this scenario",
    "source_alignment": "How this aligns with provided sources (if any)"
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
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        st.error(f"Error generating stem: {e}")
        return None

def generate_question_with_reasoning(
    client: OpenAI,
    stem: str,
    cognitive_level: str,
    source_context: str = "",
    sample_template: str = "",
    model: str = "gpt-4o",
    temperature: float = 0.4
) -> Dict[str, Any]:
    """Generate question with D&R reasoning"""
    
    source_instruction = f"\n\nSource material for validation:\n{source_context}" if source_context else ""
    
    template_instruction = f"\n\nREFERENCE: Analyze this sample item for question style and structure:\n{sample_template}" if sample_template else ""
    
    prompt = f"""
Given this clinical scenario:
{stem}

Target Cognitive Level: {cognitive_level}{source_instruction}{template_instruction}

Using Deconstruct & Reconstruct Method:

1) ANALYZE what cognitive operation the question should elicit:
   - To achieve {cognitive_level}-level thinking, what must the examinee DO mentally?
   - What decision point or reasoning challenge should the question pose?

2) RECONSTRUCT an appropriate question:
   - Frame the question to require the identified cognitive operation
   - Ensure it flows naturally from the scenario
   - Make it clear and unambiguous

Output as JSON:
{{
  "question": "What is the most appropriate...",
  "reconstruct_reasoning": {{
    "target_operation": "The specific cognitive operation (e.g., 'prioritize urgent actions', 'differentiate diagnoses')",
    "why_appropriate": "Why this question achieves {cognitive_level}-level thinking",
    "stem_connection": "How the question naturally follows from the scenario"
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
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        st.error(f"Error generating question: {e}")
        return None

def generate_options_with_reasoning(
    client: OpenAI,
    stem: str,
    question: str,
    num_options: int = 11,
    source_context: str = "",
    sample_template: str = "",
    model: str = "gpt-4o",
    temperature: float = 0.5
) -> Dict[str, Any]:
    """Generate 11 options (including 1 likely correct) using Key Feature Venn Diagram Method"""
    
    source_instruction = f"\n\nSource material for validation:\n{source_context}" if source_context else ""
    
    template_instruction = f"\n\nREFERENCE: Analyze this sample item for option style and plausibility patterns:\n{sample_template}" if sample_template else ""
    
    prompt = f"""
Using the KEY FEATURE VENN DIAGRAM METHOD to create model-based options.

Scenario: {stem}
Question: {question}{source_instruction}{template_instruction}

TASK:
1) IDENTIFY what would make an answer CORRECT:
   - List the key features a correct answer should have (e.g., time-sensitivity, diagnostic priority, safety, clinical indication)
   - Explain why each feature is critical

2) GENERATE {num_options} OPTIONS as overlapping feature sets:
   - Option 1: The MOST LIKELY correct answer (contains all critical features)
   - Options 2-{num_options}: Plausible alternatives that share SOME but not ALL critical features
   - Vary the degree of overlap (some close near-misses, some obvious errors)
   - Ensure each targets a specific misconception or incomplete reasoning

3) EXPLAIN the Venn diagram logic for EACH option:
   - Which features does it share? (overlap = plausibility)
   - Which critical feature does it lack or violate? (gap = why it might be wrong)
   - What clinical reasoning does it represent?

Output as JSON:
{{
  "key_features": ["feature 1", "feature 2", ...],
  "feature_explanations": {{
    "feature 1": "why this is critical",
    "feature 2": "why this is critical"
  }},
  "options": [
    {{
      "text": "Option text",
      "is_likely_correct": true/false,
      "shared_features": ["feature A", "feature B"],
      "violated_feature": "critical feature X (or 'none' if likely correct)",
      "venn_reasoning": "This shares [features] making it plausible, but violates [feature] because...",
      "clinical_reasoning": "What this option represents clinically"
    }},
    ... ({num_options} total)
  ]
}}
"""
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        st.error(f"Error generating options: {e}")
        return None

def regenerate_single_option(
    client: OpenAI,
    stem: str,
    question: str,
    existing_options: List[str],
    key_features: List[str],
    source_context: str = "",
    model: str = "gpt-4o",
    temperature: float = 0.6
) -> Dict[str, Any]:
    """Regenerate a single option avoiding duplicates"""
    
    prompt = f"""
Generate ONE new option using Key Feature Venn Diagram Method.

Scenario: {stem}
Question: {question}
Key Features: {', '.join(key_features)}

Existing options (avoid duplicating these):
{json.dumps(existing_options, indent=2)}

Create a NEW option that:
- Shares some key features (plausibility)
- May or may not have all critical features
- Is distinct from existing options
- Represents a different clinical reasoning path

Output as JSON:
{{
  "text": "New option text",
  "is_likely_correct": true/false,
  "shared_features": ["feature A", "feature B"],
  "violated_feature": "critical feature X (or 'none')",
  "venn_reasoning": "Explanation of feature overlap and gap",
  "clinical_reasoning": "What clinical thinking this represents"
}}
"""
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        st.error(f"Error regenerating option: {e}")
        return None

def validate_component(
    client: OpenAI,
    component_type: str,
    component_text: str,
    source_context: str = "",
    domain: str = "",
    model: str = "gpt-4o",
    temperature: float = 0.3
) -> Dict[str, Any]:
    """Validate component against sources first, then LLM knowledge"""
    
    source_priority = "PRIMARY VALIDATION SOURCE (check this first):" if source_context else "NO SOURCE PROVIDED - validate against general medical knowledge:"
    
    prompt = f"""
Validate this {component_type} for a {domain} assessment item.

{component_type.upper()}:
{component_text}

{source_priority}
{source_context if source_context else "Use standard medical knowledge and current clinical guidelines."}

VALIDATION CHECKLIST:
1) Content Accuracy:
   - Is the information factually correct?
   - Does it align with current evidence/guidelines?
   - Any contradictions with source material?

2) Source Alignment (if source provided):
   - Which specific parts align with the source?
   - Quote the relevant source text
   - Any deviations from the source?

3) Clinical Appropriateness:
   - Is this realistic for clinical practice?
   - Any safety concerns?
   - Appropriate for the specified domain?

4) Recommendations:
   - Accept as-is
   - Suggest specific revisions
   - Flag critical issues

Output as JSON:
{{
  "validation_status": "PASS" or "NEEDS_REVISION" or "FAIL",
  "content_accuracy": "Assessment",
  "source_alignment": {{
    "aligned": true/false,
    "source_quote": "Relevant quote from source (if applicable)",
    "deviations": "Any differences from source"
  }},
  "clinical_appropriateness": "Assessment",
  "recommendations": ["recommendation 1", "recommendation 2"],
  "critical_issues": ["issue 1"] or []
}}
"""
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        st.error(f"Error validating component: {e}")
        return None

# ============================================================
# Source Material Processing
# ============================================================

def extract_text_from_pdf(file) -> str:
    """Extract text from uploaded PDF"""
    if not HAS_PDF:
        return "PDF processing not available. Install PyPDF2."
    
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = []
        for page in pdf_reader.pages:
            text.append(page.extract_text())
        return "\n\n".join(text)
    except Exception as e:
        return f"Error reading PDF: {e}"

def process_uploaded_file(file) -> str:
    """Process uploaded file and extract text"""
    if file.type == "application/pdf":
        return extract_text_from_pdf(file)
    elif file.type == "text/plain":
        return file.read().decode("utf-8")
    else:
        return "Unsupported file type. Please upload PDF or TXT files."

# ============================================================
# Streamlit App
# ============================================================

def main():
    st.set_page_config(
        page_title="AIG SME Toolkit - Enhanced",
        page_icon="🧠",
        layout="wide"
    )
    
    # Custom CSS for better layout
    st.markdown("""
    <style>
    .reasoning-box {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .validation-pass {
        background-color: #d4edda;
        padding: 10px;
        border-radius: 5px;
    }
    .validation-fail {
        background-color: #f8d7da;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🧠 AIG SME Toolkit - Enhanced")
    st.markdown("**Transparent High-Order Item Generation with D&R + Key Feature Venn Diagram**")
    
    # Sidebar - Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=os.getenv("OPENAI_API_KEY", ""),
            help="Enter your OpenAI API key"
        )
        
        if not api_key:
            st.warning("⚠️ Please enter your OpenAI API key")
            st.stop()
        
        st.divider()
        
        st.subheader("Model Settings")
        model = st.selectbox("Model", ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"], index=0)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.4, 0.1)
    
    # Initialize session state
    if 'source_materials' not in st.session_state:
        st.session_state.source_materials = []
    if 'item_data' not in st.session_state:
        st.session_state.item_data = None
    if 'finalized_items' not in st.session_state:
        st.session_state.finalized_items = []
    if 'validations' not in st.session_state:
        st.session_state.validations = {}
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📚 Source Materials", "🎯 Item Creation", "🔍 Reasoning View", "📦 Finalized Items"])
    
    # ============================================================
    # TAB 1: Source Materials
    # ============================================================
    with tab1:
        st.header("📚 Import Domain Knowledge Sources")
        st.markdown("Upload reference materials to validate item content against authoritative sources.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📄 Upload Files")
            uploaded_files = st.file_uploader(
                "Upload PDF or Text files",
                type=["pdf", "txt"],
                accept_multiple_files=True,
                help="Upload textbooks, guidelines, or reference materials"
            )
            
            if uploaded_files:
                if st.button("➕ Add Uploaded Files"):
                    for file in uploaded_files:
                        content = process_uploaded_file(file)
                        st.session_state.source_materials.append({
                            'type': 'file',
                            'name': file.name,
                            'content': content[:5000],  # Limit to 5000 chars
                            'timestamp': datetime.now().isoformat()
                        })
                    st.success(f"Added {len(uploaded_files)} file(s)")
                    st.rerun()
        
        with col2:
            st.subheader("✍️ Paste Text")
            pasted_text = st.text_area(
                "Paste reference text",
                height=200,
                placeholder="Paste guidelines, protocols, or relevant content..."
            )
            
            source_name = st.text_input("Source name", placeholder="e.g., AHA Guidelines 2024")
            
            if st.button("➕ Add Pasted Text"):
                if pasted_text.strip() and source_name.strip():
                    st.session_state.source_materials.append({
                        'type': 'pasted',
                        'name': source_name,
                        'content': pasted_text[:5000],
                        'timestamp': datetime.now().isoformat()
                    })
                    st.success("Source added!")
                    st.rerun()
        
        # Display sources
        if st.session_state.source_materials:
            st.divider()
            st.subheader(f"📋 Loaded Sources ({len(st.session_state.source_materials)})")
            
            for i, source in enumerate(st.session_state.source_materials):
                with st.expander(f"{source['type'].upper()}: {source['name']}"):
                    st.text(source['content'][:500] + "..." if len(source['content']) > 500 else source['content'])
                    if st.button(f"🗑️ Remove", key=f"remove_{i}"):
                        st.session_state.source_materials.pop(i)
                        st.rerun()
        else:
            st.info("No sources loaded. Items will be validated against LLM's general knowledge only.")
    
    # ============================================================
    # TAB 2: Item Creation
    # ============================================================
    with tab2:
        client = OpenAI(api_key=api_key)
        
        # Combine all source materials
        combined_sources = "\n\n---\n\n".join([
            f"SOURCE: {s['name']}\n{s['content']}" 
            for s in st.session_state.source_materials
        ]) if st.session_state.source_materials else ""
        
        st.header("🎯 Item Specification")
        
        col1, col2 = st.columns(2)
        with col1:
            domain = st.text_input("Domain", value="Internal Medicine")
            cognitive_level = st.selectbox(
                "Target Cognitive Level",
                ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"],
                index=3
            )
        with col2:
            subtopic = st.text_input("Subtopic", value="Acute Coronary Syndrome")
            num_options = st.number_input(
                "Number of Options (including correct answer)",
                min_value=5,
                max_value=8,
                value=5,
                step=1,
                help="Total options to generate (5-8)"
            )
        
        seed_scenario = st.text_area("Seed Scenario (Optional)", height=100, 
            placeholder="Provide context or clinical situation to guide generation...")
        
        # Sample item template
        sample_item_template = st.text_area(
            "Sample Item(s) for Template (Optional)",
            height=200,
            placeholder="""Provide one or more example items to use as a template. The LLM will analyze their structure and cognitive approach.

Example format:
Stem: A 65-year-old patient presents with chest pain...
Question: What is the most appropriate initial action?
Options:
A. Obtain ECG immediately (CORRECT)
B. Schedule stress test
C. Reassure and discharge
...

This helps the LLM understand your preferred item format and cognitive complexity.""",
            help="Paste example items to guide the generation. The LLM will use these as templates for structure and cognitive level."
        )
        
        # Generate initial item
        correct_answer_input = st.text_input("Correct answer (optional - LLM will suggest one):", key="correct_answer_input")
        
        if st.button("🚀 Generate Item (Stem + Question + Options)", type="primary", use_container_width=True):
            with st.spinner(f"Generating item with {num_options} options and transparent reasoning..."):
                # Generate stem
                stem_result = generate_stem_with_reasoning(
                    client, domain, subtopic, cognitive_level, combined_sources, sample_item_template, model, temperature
                )
                
                if stem_result:
                    # Generate question
                    question_result = generate_question_with_reasoning(
                        client, stem_result['stem'], cognitive_level, combined_sources, sample_item_template, model, temperature
                    )
                    
                    if question_result:
                        # Generate options using KFVD (5-8 options)
                        options_result = generate_options_with_reasoning(
                            client, stem_result['stem'], question_result['question'], 
                            num_options, combined_sources, sample_item_template, model, temperature
                        )
                        
                        if options_result:
                            st.session_state.item_data = {
                                'stem': stem_result['stem'],
                                'stem_reasoning': stem_result['deconstruct_reasoning'],
                                'question': question_result['question'],
                                'question_reasoning': question_result['reconstruct_reasoning'],
                                'key_features': options_result['key_features'],
                                'feature_explanations': options_result['feature_explanations'],
                                'options': options_result['options'],
                                'domain': domain,
                                'subtopic': subtopic,
                                'cognitive_level': cognitive_level,
                                'num_options': num_options
                            }
                            st.success(f"✅ Item generated with {num_options} options! Review and select the correct answer.")
                            st.rerun()
        
        # Edit existing item
        if st.session_state.item_data:
            st.divider()
            st.header("✏️ Edit Item Components")
            
            item = st.session_state.item_data
            
            # STEM Section
            st.subheader("📝 Clinical Scenario (Stem)")
            col1, col2, col3 = st.columns([6, 1, 1])
            with col1:
                edited_stem = st.text_area("Stem", value=item['stem'], height=150, key="stem_edit")
            with col2:
                if st.button("🔄 Regen", key="regen_stem"):
                    with st.spinner("Regenerating stem..."):
                        stem_result = generate_stem_with_reasoning(
                            client, item['domain'], item['subtopic'], 
                            item['cognitive_level'], combined_sources, model, temperature
                        )
                        if stem_result:
                            st.session_state.item_data['stem'] = stem_result['stem']
                            st.session_state.item_data['stem_reasoning'] = stem_result['deconstruct_reasoning']
                            st.rerun()
            with col3:
                if st.button("✓ Validate", key="val_stem"):
                    with st.spinner("Validating..."):
                        val_result = validate_component(
                            client, "Clinical Scenario", edited_stem, 
                            combined_sources, item['domain'], model, temperature
                        )
                        if val_result:
                            st.session_state.validations['stem'] = val_result
                            st.rerun()
            
            # Show stem reasoning
            with st.expander("🧩 View Deconstruct & Reconstruct Reasoning for Stem"):
                reasoning = item.get('stem_reasoning', {})
                if reasoning:
                    st.markdown("**🎯 Cognitive Operations Required:**")
                    for op in reasoning.get('cognitive_operations', []):
                        st.write(f"• {op}")
                    
                    st.markdown(f"**🔢 Reasoning Steps:** {reasoning.get('reasoning_steps', 'N/A')}")
                    
                    st.markdown("**💡 Why This Cognitive Level?**")
                    st.info(reasoning.get('why_this_level', 'N/A'))
                    
                    st.markdown("**📚 Content Concepts:**")
                    for concept in reasoning.get('content_concepts', []):
                        st.write(f"• {concept}")
                    
                    st.markdown("**🔗 Concept Relationships:**")
                    st.write(reasoning.get('concept_relationships', 'N/A'))
                    
                    if reasoning.get('source_alignment'):
                        st.markdown("**📖 Source Alignment:**")
                        st.success(reasoning.get('source_alignment'))
                else:
                    st.info("No reasoning available")
            
            # Show stem validation
            if 'stem' in st.session_state.validations:
                val = st.session_state.validations['stem']
                status_class = "validation-pass" if val['validation_status'] == "PASS" else "validation-fail"
                st.markdown(f'<div class="{status_class}"><b>Validation: {val["validation_status"]}</b><br>{val.get("content_accuracy", "")}</div>', unsafe_allow_html=True)
                if val['source_alignment'].get('source_quote'):
                    st.info(f"**Source Quote:** {val['source_alignment']['source_quote']}")
            
            # QUESTION Section
            st.subheader("❓ Question")
            col1, col2, col3 = st.columns([6, 1, 1])
            with col1:
                edited_question = st.text_area("Question", value=item['question'], height=80, key="question_edit")
            with col2:
                if st.button("🔄 Regen", key="regen_question"):
                    with st.spinner("Regenerating question..."):
                        question_result = generate_question_with_reasoning(
                            client, item['stem'], item['cognitive_level'], 
                            combined_sources, model, temperature
                        )
                        if question_result:
                            st.session_state.item_data['question'] = question_result['question']
                            st.session_state.item_data['question_reasoning'] = question_result['reconstruct_reasoning']
                            st.rerun()
            with col3:
                if st.button("✓ Validate", key="val_question"):
                    with st.spinner("Validating..."):
                        val_result = validate_component(
                            client, "Question", edited_question, 
                            combined_sources, item['domain'], model, temperature
                        )
                        if val_result:
                            st.session_state.validations['question'] = val_result
                            st.rerun()
            
            with st.expander("🧩 View Reconstruct Reasoning for Question"):
                reasoning = item.get('question_reasoning', {})
                if reasoning:
                    st.markdown("**🎯 Target Mental Operation:**")
                    st.info(reasoning.get('target_operation', 'N/A'))
                    
                    st.markdown("**✅ Why This Question is Appropriate:**")
                    st.write(reasoning.get('why_appropriate', 'N/A'))
                    
                    st.markdown("**🔗 Connection to Scenario:**")
                    st.write(reasoning.get('scenario_connection', 'N/A'))
                    
                    if reasoning.get('source_alignment'):
                        st.markdown("**📖 Source Alignment:**")
                        st.success(reasoning.get('source_alignment'))
                else:
                    st.info("No reasoning available")
            
            if 'question' in st.session_state.validations:
                val = st.session_state.validations['question']
                status_class = "validation-pass" if val['validation_status'] == "PASS" else "validation-fail"
                st.markdown(f'<div class="{status_class}"><b>Validation: {val["validation_status"]}</b><br>{val.get("content_accuracy", "")}</div>', unsafe_allow_html=True)
            
            # OPTIONS Section
            st.subheader(f"🎯 All Options ({len(item.get('options', []))} Total)")
            
            if 'options' in item and item['options']:
                # Initialize selection states if not exists
                if 'selected_correct' not in st.session_state:
                    # Find LLM's suggested correct answer
                    suggested_idx = next((i for i, opt in enumerate(item['options']) if opt.get('is_likely_correct')), 0)
                    st.session_state.selected_correct = suggested_idx
                
                if 'selected_options' not in st.session_state:
                    # By default, select all options for inclusion
                    st.session_state.selected_options = set(range(len(item['options'])))
                
                st.info(f"ℹ️ **🔘 = Mark as CORRECT | ☑️ = Include in final item**")
                
                # Display all options
                for i, opt in enumerate(item['options']):
                    option_label = chr(65 + i)  # A, B, C, D, E, F, G, H, I, J, K
                    is_selected_correct = (i == st.session_state.selected_correct)
                    is_included = i in st.session_state.selected_options
                    
                    # Create columns for correct button, include checkbox, content, and action buttons
                    col_correct, col_include, col_content, col_regen, col_validate = st.columns([0.5, 0.5, 4.5, 0.8, 0.8])
                    
                    with col_correct:
                        st.write("")  # Spacer
                        st.write("")  # Spacer
                        # Button to mark as correct answer
                        if st.button("⚪" if not is_selected_correct else "🔘", key=f"select_{i}", help="Mark as correct answer"):
                            st.session_state.selected_correct = i
                            # Auto-include if marked as correct
                            st.session_state.selected_options.add(i)
                            st.rerun()
                    
                    with col_include:
                        st.write("")  # Spacer
                        st.write("")  # Spacer
                        # Checkbox to include/exclude option
                        include_label = "☑️" if is_included else "☐"
                        if st.button(include_label, key=f"include_{i}", help="Include/exclude this option"):
                            if i in st.session_state.selected_options:
                                # Don't allow excluding the correct answer
                                if i == st.session_state.selected_correct:
                                    st.error("Cannot exclude the correct answer!")
                                else:
                                    st.session_state.selected_options.remove(i)
                                    st.rerun()
                            else:
                                st.session_state.selected_options.add(i)
                                st.rerun()
                    
                    with col_content:
                        # Mark the selected correct answer and inclusion status
                        prefix = "✅ **CORRECT** - " if is_selected_correct else ""
                        llm_suggestion = " 🤖 (LLM suggested)" if opt.get('is_likely_correct') else ""
                        excluded_marker = " ❌ (EXCLUDED)" if not is_included else ""
                        st.markdown(f"**{prefix}Option {option_label}**{llm_suggestion}{excluded_marker}")
                        
                        opt['text'] = st.text_area(
                            f"Option {option_label}", 
                            value=opt['text'], 
                            height=80, 
                            key=f"opt_{i}",
                            label_visibility="collapsed",
                            disabled=not is_included  # Disable editing if excluded
                        )
                    
                    with col_regen:
                        st.write("")  # Spacer
                        st.write("")  # Spacer
                        if st.button("🔄", key=f"regen_opt_{i}", help="Regenerate this option"):
                            with st.spinner(f"Regenerating option {option_label}..."):
                                existing = [o['text'] for j, o in enumerate(item['options']) if j != i]
                                new_opt = regenerate_single_option(
                                    client, item['stem'], item['question'], 
                                    existing, item['key_features'], 
                                    combined_sources, model, temperature
                                )
                                if new_opt:
                                    st.session_state.item_data['options'][i] = new_opt
                                    st.rerun()
                    
                    with col_validate:
                        st.write("")  # Spacer
                        st.write("")  # Spacer
                        if st.button("✓", key=f"val_opt_{i}", help="Validate against sources"):
                            with st.spinner("Validating..."):
                                val_result = validate_component(
                                    client, f"Option {option_label}", opt['text'], 
                                    combined_sources, item['domain'], model, temperature
                                )
                                if val_result:
                                    st.session_state.validations[f'opt_{i}'] = val_result
                                    st.rerun()
                    
                    # Show Venn diagram reasoning
                    with st.expander(f"🔍 Key Feature Venn Diagram Reasoning - Option {option_label}"):
                        if opt.get('is_likely_correct'):
                            st.success("🎯 **LLM suggests this as the likely CORRECT answer**")
                        st.write(f"**Shared Features (Plausibility):** {', '.join(opt.get('shared_features', []))}")
                        st.write(f"**Violated Feature:** {opt.get('violated_feature', 'None')}")
                        st.write(f"**Venn Reasoning:** {opt.get('venn_reasoning', 'N/A')}")
                        st.write(f"**Clinical Reasoning:** {opt.get('clinical_reasoning', 'N/A')}")
                    
                    # Show validation
                    if f'opt_{i}' in st.session_state.validations:
                        val = st.session_state.validations[f'opt_{i}']
                        status_class = "validation-pass" if val['validation_status'] == "PASS" else "validation-fail"
                        with col_content:
                            st.markdown(f'<div class="{status_class}">Validation: {val["validation_status"]}</div>', unsafe_allow_html=True)
                            if val['source_alignment'].get('source_quote'):
                                st.info(f"**Source:** {val['source_alignment']['source_quote'][:200]}...")
                
                # Summary of selection
                st.divider()
                selected_opt = item['options'][st.session_state.selected_correct]
                included_count = len(st.session_state.selected_options)
                st.success(f"✅ **Selected Correct Answer:** Option {chr(65 + st.session_state.selected_correct)} - {selected_opt['text'][:100]}...")
                st.info(f"📊 **Included Options:** {included_count} of {len(item['options'])} options will be in the final item")
                
                # Show key features
                with st.expander("🔑 View Key Features (What makes an answer correct)"):
                    st.markdown("**Critical features that the correct answer must possess:**")
                    st.markdown("")  # Spacing
                    
                    key_features = item.get('key_features', [])
                    feature_explanations = item.get('feature_explanations', {})
                    
                    if key_features:
                        for idx, feat in enumerate(key_features, 1):
                            explain = feature_explanations.get(feat, "")
                            
                            # Display each feature with icon and explanation
                            st.markdown(f"**{idx}. ✓ {feat.title().replace('-', ' ').replace('_', ' ')}**")
                            if explain:
                                st.info(f"💡 {explain}")
                            else:
                                st.caption("(No explanation provided)")
                            st.markdown("")  # Spacing between features
                    else:
                        st.warning("No key features defined yet.")
            else:
                st.warning("No options generated yet. Generate an item first.")
            
            # Finalize
            st.divider()
            if st.button("✅ Finalize Item", type="primary", use_container_width=True):
                if 'options' not in item or not item['options']:
                    st.error("⚠️ No options to finalize. Generate an item first.")
                elif len(st.session_state.selected_options) < 2:
                    st.error("⚠️ Please include at least 2 options (1 correct + 1 distractor).")
                else:
                    correct_idx = st.session_state.get('selected_correct', 0)
                    # Filter to only included options and renumber them
                    included_indices = sorted(list(st.session_state.selected_options))
                    included_options = [item['options'][i] for i in included_indices]
                    # Find new index of correct answer in filtered list
                    new_correct_idx = included_indices.index(correct_idx)
                    
                    finalized = {
                        'item_id': str(uuid.uuid4()),
                        'timestamp': datetime.now().isoformat(),
                        'domain': item['domain'],
                        'subtopic': item['subtopic'],
                        'cognitive_level': item['cognitive_level'],
                        'stem': edited_stem,
                        'stem_reasoning': item.get('stem_reasoning', {}),
                        'question': edited_question,
                        'question_reasoning': item.get('question_reasoning', {}),
                        'options': included_options,
                        'correct_index': new_correct_idx,
                        'correct_answer': item['options'][correct_idx]['text'],
                        'key_features': item.get('key_features', []),
                        'feature_explanations': item.get('feature_explanations', {}),
                        'validations': st.session_state.validations,
                        'sources_used': [s['name'] for s in st.session_state.source_materials]
                    }
                    st.session_state.finalized_items.append(finalized)
                    st.success(f"✅ Item finalized with {len(included_options)} options! Total: {len(st.session_state.finalized_items)}")
                    # Clear selections for next item
                    st.session_state.selected_correct = 0
                    st.session_state.selected_options = set()
    
    # ============================================================
    # TAB 3: Reasoning View
    # ============================================================
    with tab3:
        st.header("🔍 Modeling Approaches - Transparent Reasoning")
        
        if st.session_state.item_data:
            item = st.session_state.item_data
            
            st.subheader("1️⃣ Deconstruct & Reconstruct Method")
            st.markdown("**Rationale**: Creates cognitively isomorphic items by explicitly modeling mental operations and content integration required.")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**🧩 Stem Deconstruction**")
                if 'stem_reasoning' in item:
                    reasoning = item['stem_reasoning']
                    
                    st.markdown("**🎯 Cognitive Operations:**")
                    for op in reasoning.get('cognitive_operations', []):
                        st.write(f"• {op}")
                    
                    st.markdown(f"**🔢 Steps:** {reasoning.get('reasoning_steps', 'N/A')}")
                    
                    st.markdown("**💡 Cognitive Level:**")
                    st.info(reasoning.get('why_this_level', 'N/A'))
                    
                    st.markdown("**📚 Content:**")
                    for concept in reasoning.get('content_concepts', []):
                        st.write(f"• {concept}")
                    
                    st.markdown("**🔗 Relationships:**")
                    st.caption(reasoning.get('concept_relationships', 'N/A'))
                else:
                    st.info("Generate stem to view reasoning")
            
            with col2:
                st.markdown("**🧩 Question Reconstruction**")
                if 'question_reasoning' in item:
                    reasoning = item['question_reasoning']
                    
                    st.markdown("**🎯 Target Operation:**")
                    st.info(reasoning.get('target_operation', 'N/A'))
                    
                    st.markdown("**✅ Appropriateness:**")
                    st.write(reasoning.get('why_appropriate', 'N/A'))
                    
                    st.markdown("**🔗 Scenario Connection:**")
                    st.caption(reasoning.get('scenario_connection', 'N/A'))
                else:
                    st.info("Generate question to view reasoning")
            
            st.divider()
            
            st.subheader("2️⃣ Key Feature Venn Diagram Method")
            st.markdown("**Rationale**: Creates plausible distractors by modeling overlapping feature sets - each distractor shares some (but not all) critical features with the correct answer.")
            
            st.markdown("**🔑 Correct Answer Key Features**")
            if 'key_features' in item:
                for feat in item['key_features']:
                    explain = item.get('feature_explanations', {}).get(feat, "")
                    st.write(f"- **{feat}**: {explain}")
            
            st.markdown("**🎭 Option Feature Analysis**")
            for i, opt in enumerate(item.get('options', [])):
                is_correct = (i == st.session_state.get('selected_correct', 0))
                label = "✅ CORRECT" if is_correct else "Distractor"
                with st.expander(f"{label} - Option {chr(65+i)}: {opt['text'][:50]}..."):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**✅ Shared Features (Plausibility)**")
                        for feat in opt.get('shared_features', []):
                            st.write(f"- {feat}")
                    with col2:
                        st.markdown("**❌ Violated Feature**")
                        violated = opt.get('violated_feature', 'None')
                        if violated == 'None' or violated == 'none':
                            st.success("No critical feature violated (likely correct)")
                        else:
                            st.error(violated)
                    
                    st.markdown("**💡 Venn Diagram Logic**")
                    st.info(opt.get('venn_reasoning', 'N/A'))
                    
                    st.markdown("**🎯 Clinical Reasoning**")
                    st.write(opt.get('clinical_reasoning', 'N/A'))
        else:
            st.info("Generate an item in the 'Item Creation' tab to view reasoning transparency.")
    
    # ============================================================
    # TAB 4: Finalized Items
    # ============================================================
    with tab4:
        st.header("📦 Finalized Items")
        
        # Initialize feedback storage if not exists
        if 'item_feedback' not in st.session_state:
            st.session_state.item_feedback = {}
        
        if st.session_state.finalized_items:
            st.write(f"**Total Items:** {len(st.session_state.finalized_items)}")
            
            for i, item in enumerate(st.session_state.finalized_items):
                with st.expander(f"📝 Item {i+1}: {item['subtopic']} ({item['cognitive_level']})", expanded=False):
                    # Header Information
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write(f"**Item ID:** `{item['item_id'][:8]}...`")
                        st.write(f"**Domain:** {item['domain']}")
                    with col2:
                        st.write(f"**Subtopic:** {item['subtopic']}")
                        st.write(f"**Cognitive Level:** {item['cognitive_level']}")
                    with col3:
                        st.write(f"**Created:** {item['timestamp'][:10]}")
                        st.write(f"**Options:** {len(item['options'])}")
                    
                    st.divider()
                    
                    # Clinical Scenario (Stem)
                    st.markdown("### 📋 Clinical Scenario")
                    st.write(item['stem'])
                    
                    # D&R Reasoning for Stem
                    if 'stem_reasoning' in item and item['stem_reasoning']:
                        with st.expander("🧩 Deconstruct & Reconstruct Rationale - Stem"):
                            reasoning = item['stem_reasoning']
                            
                            st.markdown("**🎯 Cognitive Operations Required:**")
                            for op in reasoning.get('cognitive_operations', []):
                                st.write(f"- {op}")
                            
                            st.markdown(f"**🔢 Reasoning Steps:** {reasoning.get('reasoning_steps', 'N/A')}")
                            
                            st.markdown("**💡 Why This Cognitive Level?**")
                            st.info(reasoning.get('why_this_level', 'N/A'))
                            
                            st.markdown("**📚 Content Concepts:**")
                            for concept in reasoning.get('content_concepts', []):
                                st.write(f"- {concept}")
                            
                            st.markdown("**🔗 Concept Relationships:**")
                            st.write(reasoning.get('concept_relationships', 'N/A'))
                            
                            if reasoning.get('source_alignment'):
                                st.markdown("**📖 Source Alignment:**")
                                st.success(reasoning.get('source_alignment'))
                    
                    st.divider()
                    
                    # Question
                    st.markdown("### ❓ Question")
                    st.write(item['question'])
                    
                    # D&R Reasoning for Question
                    if 'question_reasoning' in item and item['question_reasoning']:
                        with st.expander("🧩 Deconstruct & Reconstruct Rationale - Question"):
                            reasoning = item['question_reasoning']
                            
                            st.markdown("**🎯 Target Mental Operation:**")
                            st.info(reasoning.get('target_operation', 'N/A'))
                            
                            st.markdown("**✅ Why This Question is Appropriate:**")
                            st.write(reasoning.get('why_appropriate', 'N/A'))
                            
                            st.markdown("**🔗 Connection to Scenario:**")
                            st.write(reasoning.get('stem_connection', 'N/A'))
                    
                    st.divider()
                    
                    # Options
                    st.markdown("### 🎯 Options")
                    
                    for opt_idx, opt in enumerate(item['options']):
                        is_correct = (opt_idx == item['correct_index'])
                        option_letter = chr(65 + opt_idx)
                        
                        if is_correct:
                            st.success(f"**✅ Option {option_letter} (CORRECT ANSWER)**")
                        else:
                            st.markdown(f"**Option {option_letter}**")
                        
                        st.write(opt['text'])
                        
                        # KFVD Reasoning
                        with st.expander(f"🔍 Key Feature Venn Diagram Analysis - Option {option_letter}"):
                            if is_correct:
                                st.markdown("**🎯 This is the CORRECT answer**")
                            
                            st.markdown("**✅ Features Present (Makes it Plausible):**")
                            for feat in opt.get('shared_features', []):
                                st.write(f"✓ {feat}")
                            
                            violated = opt.get('violated_feature', 'None')
                            if violated and violated.lower() != 'none':
                                st.markdown("**❌ Critical Feature Violated (Makes it Incorrect):**")
                                st.error(violated)
                            else:
                                st.markdown("**✅ All Critical Features Present:**")
                                st.success("No features violated - this is the correct answer")
                            
                            st.markdown("**💭 Clinical Reasoning:**")
                            st.info(opt.get('venn_reasoning', 'N/A'))
                            
                            st.markdown("**🎓 What This Tests:**")
                            st.write(opt.get('clinical_reasoning', 'N/A'))
                    
                    st.divider()
                    
                    # Key Features Summary
                    st.markdown("### 🔑 Key Features (What Makes an Answer Correct)")
                    
                    if 'key_features' in item:
                        for feat in item['key_features']:
                            explanation = item.get('feature_explanations', {}).get(feat, '')
                            if explanation:
                                st.write(f"**{feat}:** {explanation}")
                            else:
                                st.write(f"- {feat}")
                    
                    st.divider()
                    
                    # Metadata
                    st.markdown("### 📊 Item Metadata")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Sources Used:**")
                        if item['sources_used']:
                            for source in item['sources_used']:
                                st.write(f"- {source}")
                        else:
                            st.write("_No specific sources_")
                    
                    with col2:
                        st.markdown("**Validations Performed:**")
                        if item.get('validations'):
                            st.write(f"✓ {len(item['validations'])} components validated")
                        else:
                            st.write("_No validations recorded_")
                    
                    st.divider()
                    
                    # ======================================
                    # SME FEEDBACK FOR FINE-TUNING
                    # ======================================
                    st.markdown("### 📝 SME Feedback (for LLM Fine-Tuning)")
                    st.caption("Your feedback helps improve the AI model's item generation over time")
                    
                    # Initialize feedback for this item if not exists
                    item_id = item['item_id']
                    if item_id not in st.session_state.item_feedback:
                        st.session_state.item_feedback[item_id] = {
                            'rating': None,
                            'feedback_text': '',
                            'timestamp': None
                        }
                    
                    feedback_col1, feedback_col2 = st.columns([1, 4])
                    
                    with feedback_col1:
                        st.markdown("**Overall Quality:**")
                        
                        # Thumbs Up/Down buttons
                        thumb_col1, thumb_col2, thumb_col3 = st.columns(3)
                        
                        with thumb_col1:
                            current_rating = st.session_state.item_feedback[item_id]['rating']
                            thumbs_up_label = "👍" if current_rating != 'up' else "👍✓"
                            
                            if st.button(thumbs_up_label, key=f"thumbs_up_{i}", help="Good quality item"):
                                st.session_state.item_feedback[item_id]['rating'] = 'up'
                                st.session_state.item_feedback[item_id]['timestamp'] = datetime.now().isoformat()
                                st.rerun()
                        
                        with thumb_col2:
                            thumbs_down_label = "👎" if current_rating != 'down' else "👎✓"
                            
                            if st.button(thumbs_down_label, key=f"thumbs_down_{i}", help="Needs improvement"):
                                st.session_state.item_feedback[item_id]['rating'] = 'down'
                                st.session_state.item_feedback[item_id]['timestamp'] = datetime.now().isoformat()
                                st.rerun()
                        
                        with thumb_col3:
                            if current_rating:
                                if st.button("🔄", key=f"clear_rating_{i}", help="Clear rating"):
                                    st.session_state.item_feedback[item_id]['rating'] = None
                                    st.rerun()
                    
                    with feedback_col2:
                        st.markdown("**Detailed Feedback (Optional):**")
                        
                        feedback_text = st.text_area(
                            "What could be improved?",
                            value=st.session_state.item_feedback[item_id]['feedback_text'],
                            height=100,
                            key=f"feedback_text_{i}",
                            placeholder="E.g., 'Options are too similar', 'Stem lacks clinical context', 'Cognitive level mismatch', 'Distractor X is implausible'...",
                            label_visibility="collapsed"
                        )
                        
                        # Save feedback text when it changes
                        if feedback_text != st.session_state.item_feedback[item_id]['feedback_text']:
                            st.session_state.item_feedback[item_id]['feedback_text'] = feedback_text
                            st.session_state.item_feedback[item_id]['timestamp'] = datetime.now().isoformat()
                    
                    # Show current feedback status
                    if current_rating or st.session_state.item_feedback[item_id]['feedback_text']:
                        rating_emoji = "👍" if current_rating == 'up' else "👎" if current_rating == 'down' else ""
                        st.success(f"{rating_emoji} Feedback saved • {st.session_state.item_feedback[item_id]['timestamp'][:16] if st.session_state.item_feedback[item_id]['timestamp'] else ''}")
            
            st.divider()
            
            # Fine-Tuning Methodology Info
            with st.expander("📚 How to Use Feedback for LLM Fine-Tuning & Continuous Improvement"):
                st.markdown("""
                ### 🎯 Fine-Tuning Methodology for Item Generation
                
                #### **What is Fine-Tuning?**
                Fine-tuning is the process of taking a pre-trained LLM (like GPT-4) and further training it on domain-specific examples 
                to improve its performance for specialized tasks—in this case, generating high-quality NCLEX-style items.
                
                #### **How SME Feedback Powers Continuous Improvement:**
                
                **1. Feedback Collection (This Tool)**
                - **👍 Thumbs Up**: Marks high-quality items as positive training examples
                - **👎 Thumbs Down**: Identifies items that need improvement
                - **Text Feedback**: Captures specific issues (e.g., "distractors too obvious", "stem lacks clinical realism")
                
                **2. Data Preparation for Fine-Tuning**
                - Export items with feedback ratings as training dataset
                - **Positive Examples** (👍): Used as "good" examples showing desired output
                - **Negative Examples** (👎): Either excluded or used with corrected versions
                - Structure data in OpenAI's fine-tuning format (JSONL):
                  ```json
                  {"messages": [
                    {"role": "system", "content": "You are an expert NCLEX item writer..."},
                    {"role": "user", "content": "Generate a stem for: Domain=Cardiology, Level=Analyze..."},
                    {"role": "assistant", "content": "A 68-year-old patient presents..."}
                  ]}
                  ```
                
                **3. Training Process**
                - Upload prepared dataset to OpenAI Fine-Tuning API
                - Model learns patterns from your approved items
                - Recommended: **50-100+ high-quality examples** per domain/cognitive level
                - Fine-tuned models can be versioned (v1, v2, v3...)
                
                **4. Iterative Improvement Cycle**
                ```
                Generate Items → SME Review → Collect Feedback → 
                Export Training Data → Fine-Tune Model → 
                Use Fine-Tuned Model → (Repeat)
                ```
                
                #### **Best Practices:**
                
                **Quality Over Quantity**
                - 50 excellent examples > 500 mediocre ones
                - Ensure approved items truly represent gold standard
                
                **Balanced Dataset**
                - Include diverse cognitive levels (Remember through Create)
                - Cover multiple domains and subtopics
                - Mix of stem types (acute, chronic, preventive, etc.)
                
                **Specific Feedback**
                - Instead of "bad item" → "Distractor B shares too many key features with correct answer"
                - Helps identify what to fix before using as training data
                
                **Version Control**
                - Track which model version generated each item
                - Compare performance across versions
                - Roll back if newer model performs worse
                
                #### **Using Feedback Data:**
                
                **Immediate Use (Manual)**
                - Review 👎 items to identify common failure patterns
                - Adjust prompts/instructions based on feedback themes
                - Regenerate poor items with refined guidance
                
                **Long-Term Use (Fine-Tuning)**
                - Accumulate 100+ rated items per specialty area
                - Export feedback dataset (see download buttons below)
                - Use OpenAI Fine-Tuning API or similar platforms
                - Deploy fine-tuned model back into this tool
                
                #### **Expected Improvements After Fine-Tuning:**
                - Better alignment with your institution's item writing style
                - Fewer implausible distractors
                - More clinically realistic scenarios
                - Improved cognitive level targeting
                - Reduced need for manual editing
                
                #### **Resources:**
                - [OpenAI Fine-Tuning Guide](https://platform.openai.com/docs/guides/fine-tuning)
                - [Best Practices for Fine-Tuning](https://platform.openai.com/docs/guides/fine-tuning/preparing-your-dataset)
                """)
            
            st.divider()
            
            # Download
            st.subheader("📥 Export Options")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Full JSON with reasoning
                json_data = json.dumps(st.session_state.finalized_items, indent=2, ensure_ascii=False)
                st.download_button(
                    "📥 Items + Reasoning",
                    data=json_data,
                    file_name=f"items_with_reasoning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True,
                    help="Download items with full D&R and KFVD reasoning"
                )
            
            with col2:
                # Items + Feedback for fine-tuning
                items_with_feedback = []
                for item in st.session_state.finalized_items:
                    item_copy = item.copy()
                    item_id = item['item_id']
                    if item_id in st.session_state.item_feedback:
                        item_copy['sme_feedback'] = st.session_state.item_feedback[item_id]
                    items_with_feedback.append(item_copy)
                
                feedback_json = json.dumps(items_with_feedback, indent=2, ensure_ascii=False)
                st.download_button(
                    "📥 Items + SME Feedback",
                    data=feedback_json,
                    file_name=f"items_with_feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True,
                    help="Download items with SME ratings and comments for fine-tuning"
                )
            
            with col3:
                # Training data format (JSONL for OpenAI)
                training_examples = []
                for item in st.session_state.finalized_items:
                    item_id = item['item_id']
                    feedback = st.session_state.item_feedback.get(item_id, {})
                    
                    # Only include items with positive feedback for training
                    if feedback.get('rating') == 'up':
                        # Format for fine-tuning: user prompt -> assistant response
                        training_examples.append({
                            "messages": [
                                {
                                    "role": "system",
                                    "content": "You are an expert NCLEX item writer using Deconstruct & Reconstruct and Key Feature Venn Diagram methods."
                                },
                                {
                                    "role": "user",
                                    "content": f"Generate a complete NCLEX item:\nDomain: {item['domain']}\nSubtopic: {item['subtopic']}\nCognitive Level: {item['cognitive_level']}\nNumber of Options: {len(item['options'])}"
                                },
                                {
                                    "role": "assistant",
                                    "content": json.dumps({
                                        "stem": item['stem'],
                                        "question": item['question'],
                                        "options": item['options'],
                                        "correct_index": item['correct_index']
                                    }, ensure_ascii=False)
                                }
                            ]
                        })
                
                if training_examples:
                    training_jsonl = "\n".join([json.dumps(ex, ensure_ascii=False) for ex in training_examples])
                    st.download_button(
                        f"📥 Training Data ({len(training_examples)} 👍)",
                        data=training_jsonl,
                        file_name=f"finetuning_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl",
                        mime="application/jsonl",
                        use_container_width=True,
                        help="OpenAI fine-tuning format (only 👍 items)"
                    )
                else:
                    st.button(
                        "📥 Training Data (0 👍)",
                        disabled=True,
                        use_container_width=True,
                        help="No items with positive feedback yet"
                    )
            
            st.divider()
            
            # Summary Statistics
            col1, col2, col3, col4 = st.columns(4)
            
            total_items = len(st.session_state.finalized_items)
            thumbs_up = sum(1 for fid in st.session_state.item_feedback.values() if fid.get('rating') == 'up')
            thumbs_down = sum(1 for fid in st.session_state.item_feedback.values() if fid.get('rating') == 'down')
            with_comments = sum(1 for fid in st.session_state.item_feedback.values() if fid.get('feedback_text'))
            
            with col1:
                st.metric("Total Items", total_items)
            with col2:
                st.metric("👍 Approved", thumbs_up)
            with col3:
                st.metric("👎 Needs Work", thumbs_down)
            with col4:
                st.metric("💬 With Comments", with_comments)
            
            st.divider()
            
            # Clear button
            # Clear button
            if st.button("🗑️ Clear All Items & Feedback", type="secondary", use_container_width=True):
                st.session_state.finalized_items = []
                st.session_state.item_feedback = {}
                st.rerun()
        else:
            st.info("No finalized items yet. Create and finalize items in the 'Item Creation' tab.")
    
    # Footer
    st.divider()
    st.markdown("""
    **AIG SME Toolkit - Enhanced** | Powered by OpenAI GPT-4o  
    *Transparent reasoning with Deconstruct & Reconstruct + Key Feature Venn Diagram*
    """)

if __name__ == "__main__":
    main()
