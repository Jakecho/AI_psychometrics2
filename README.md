# NCLEX Item Generator & Vector Search System

A comprehensive suite of AI-powered tools for generating, managing, and searching NCLEX-RN test items.

## 🎯 Applications

### 1. **AIG_NCLEX.py** - AI Item Generator
Generate high-quality NCLEX-style test items using OpenAI's GPT models with automatic quality validation.

**Features:**
- 🤖 AI-powered item generation with GPT-4o/GPT-4o-mini
- 🔀 Random answer key assignment (prevents pattern bias)
- 🔍 Automatic quality evaluation (grammar, NCLEX guidelines, bias detection)
- 🔄 Auto-regeneration with feedback for problematic items
- 📊 Supports all NCLEX client needs categories
- 📝 Customizable difficulty, cognitive levels, and focus areas
- 💾 Export to CSV and JSON formats

### 2. **NCLEX_RN_VectorDB.py** - Vector Search System
Semantic search engine for finding similar NCLEX items using pgvector and sentence transformers.

**Features:**
- 🔎 Three search methods: Pure Semantic, Weighted Hybrid, RRF
- 🎯 Flexible filtering: Top K only, Top P only, or Both
- 📊 Dynamic similarity threshold adjustment
- 🎨 Color-coded answer display
- 📈 1,704 unique items with full embeddings
- 🔍 Search by text query or item ID lookup
- 💾 Export results to CSV/JSON

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.13+ recommended
python --version

# PostgreSQL with pgvector extension
psql --version
```

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/Jakecho/nclex-item-generator.git
cd nclex-item-generator
```

2. **Create virtual environment:**
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows PowerShell
# OR
source .venv/bin/activate    # Linux/Mac
```

3. **Install dependencies:**
```bash
pip install streamlit openai psycopg2-binary sentence-transformers pandas numpy tqdm
```

4. **Set up OpenAI API Key:**
```bash
# Windows PowerShell
$env:OPENAI_API_KEY = "sk-your-api-key-here"

# Linux/Mac
export OPENAI_API_KEY="sk-your-api-key-here"
```

5. **Set up PostgreSQL database (for Vector Search):**
```sql
CREATE DATABASE pgvector;
\c pgvector
CREATE EXTENSION vector;

CREATE TABLE itembank (
    item_id VARCHAR(50) PRIMARY KEY,
    domain VARCHAR(200),
    topic VARCHAR(200),
    stem TEXT,
    "choice_A" TEXT,
    "choice_B" TEXT,
    "choice_C" TEXT,
    "choice_D" TEXT,
    key VARCHAR(1),
    rationale TEXT,
    rasch_b FLOAT,
    pvalue FLOAT,
    point_biserial FLOAT,
    combined TEXT,
    embedding vector(384)
);
```

## 📖 Usage

### Running the AI Item Generator

```bash
streamlit run AIG_NCLEX.py
```

**Access:** http://localhost:8501

**Example Workflow:**
1. Enter OpenAI API key in sidebar
2. Select domain (e.g., "Physiological Adaptation")
3. Choose NCLEX focus (e.g., "priority/immediate intervention")
4. Set difficulty level and cognitive level
5. Enter focus statement and clinical context
6. Click "🚀 Generate NCLEX Items"
7. Items are automatically:
   - Generated with randomized answer keys
   - Evaluated for quality issues
   - Regenerated if problems detected
8. Download results as CSV or JSON

### Running the Vector Search System

```bash
streamlit run NCLEX_RN_VectorDB.py
```

**Access:** http://localhost:8501

**Search Methods:**
- **Pure Semantic:** Vector similarity only (best for conceptual matches)
- **Weighted Hybrid:** Combines keywords + semantic (adjustable weights)
- **RRF:** Reciprocal Rank Fusion (parameter-free)

**Filtering Options:**
- **Top K only:** Get top N results
- **Top P only:** Get all results above similarity threshold
- **Both:** Combine K and P filtering

## 📊 Database Management

### Generate Embeddings for All Items

Use the Jupyter notebook:
```bash
jupyter notebook pgvector.ipynb
```

Or run the standalone script:
```bash
python generate_all_embeddings.py
```

### Check Database Statistics

```bash
python check_database_stats.py
```

### Clean Duplicates

```bash
python cleanup_duplicates.py
```

## 🎓 NCLEX Client Needs Categories

The system supports all 8 NCLEX categories:

1. **Management of Care**
2. **Safety and Infection Control**
3. **Health Promotion and Maintenance**
4. **Psychosocial Integrity**
5. **Basic Care and Comfort**
6. **Pharmacological and Parenteral Therapies**
7. **Reduction of Risk Potential**
8. **Physiological Adaptation**

## 📝 Sample Item Specifications

### Physiological Adaptation - Respiratory Distress

```
Client needs category: Physiological Adaptation
NCLEX focus: priority/immediate intervention
Cognitive level: analysis
Difficulty: moderate

Focus statement:
The nurse must recognize signs of acute respiratory distress and prioritize 
immediate interventions to maintain airway patency and adequate oxygenation, 
following the ABC (airway, breathing, circulation) priority framework.

Client situation:
An adult client with pneumonia experiencing increased work of breathing, 
decreased oxygen saturation, and respiratory distress requiring immediate 
nursing assessment and intervention.

Common pitfalls:
Delaying emergency interventions to complete documentation; choosing comfort 
measures over life-saving airway interventions; failing to recognize severity 
of respiratory compromise.
```

### Pharmacological Therapies - Medication Safety

```
Client needs category: Pharmacological and Parenteral Therapies
NCLEX focus: medication administration/safe practice
Cognitive level: application
Difficulty: moderate

Focus statement:
The nurse must apply safe medication administration practices, including 
verifying high-alert medications, understanding adverse effects, and 
implementing appropriate monitoring for clients receiving IV medications.

Client situation:
Multiple clients on a medical unit requiring various IV medications, 
including high-alert drugs, with the nurse needing to prioritize safe 
administration and monitoring.

Common pitfalls:
Administering medications without double-checking high-alert drugs; 
skipping patient identification; ignoring contraindications or drug interactions.
```

## 🛠️ Technical Details

### Vector Search Architecture

- **Embedding Model:** all-MiniLM-L6-v2 (384 dimensions)
- **Database:** PostgreSQL with pgvector extension
- **Distance Metric:** Cosine similarity
- **Search Algorithms:** Pure semantic, weighted hybrid, RRF
- **Dataset:** 1,704 unique NCLEX items (fully embedded)

### AI Generation

- **Models Supported:** GPT-4o, GPT-4o-mini, GPT-5-mini
- **Token Limit:** 16,000 tokens (supports multiple items)
- **Temperature:** 0.3 (balanced creativity/consistency)
- **Quality Checks:** Grammar, NCLEX guidelines, bias detection
- **Max Regenerations:** 2 attempts per item

## 📄 File Structure

```
.
├── AIG_NCLEX.py                    # AI item generator
├── NCLEX_RN_VectorDB.py            # Vector search system
├── pgvector.ipynb                  # Embedding generation notebook
├── generate_all_embeddings.py      # Batch embedding script
├── cleanup_duplicates.py           # Database deduplication
├── check_database_stats.py         # Database statistics
├── hybrid_search_demo.py           # Search method demonstration
├── pgvector_search_options.md      # Search strategy documentation
├── item_bank_clean.csv             # Original item bank
├── NCLEX_style_item_bank.csv       # Formatted item bank
└── README.md                       # This file
```

## 🔒 Security

- OpenAI API key required (not included)
- Password-protected application access
- Local database (no external data sharing)

## 📚 Documentation

- **Search Strategies:** See `pgvector_search_options.md`
- **API Reference:** OpenAI GPT-4 documentation
- **Database Schema:** See installation section above

## 🤝 Contributing

This project was developed for the ICE Exchange 2025 Conference.

## 📧 Contact

**Author:** Jake Cho  
**Repository:** https://github.com/Jakecho/nclex-item-generator  
**Branch:** main

## 📄 License

Educational use only. Not for commercial distribution.

## 🙏 Acknowledgments

- OpenAI for GPT models
- pgvector for PostgreSQL vector similarity search
- Sentence Transformers for embedding models
- Streamlit for the web interface framework

---

**Version:** 1.0.0  
**Last Updated:** November 16, 2025  
**Conference:** ICE Exchange 2025
