# RAG Training Program

Comprehensive training and testing system for the Mar Vista Librarian RAG (Retrieval-Augmented Generation) search engine.

## 🎯 Overview

This training program creates an automated feedback loop to continuously improve the RAG system's search accuracy and topology understanding. It includes:

1. **Deep Topology Analyzer** - Enhanced extraction of structural elements using GPT-4 Vision
2. **Scenario Generator** - Creates comprehensive test scenarios covering 90% of knowledge base
3. **Test Simulator** - Executes searches, validates results, generates improvement feedback
4. **Training Dashboard** - Web UI for visualizing results and tracking progress
5. **Human-in-the-Loop** - Manual checkpoints at each critical step

## 📋 Requirements

- Python 3.10+
- OpenAI API key with GPT-4 and GPT-4 Vision access
- Running API server (`python api.py`)
- Mar Vista project already indexed in the knowledge base

## 🚀 Quick Start

### Option 1: Interactive Menu (Recommended)

```bash
python training_controller.py
```

This launches an interactive menu where you can:
- Run the full pipeline
- Execute individual steps
- Check pipeline status
- View command reference

### Option 2: Full Automated Pipeline

```bash
python training_controller.py full
```

Runs all 4 steps sequentially with human-in-the-loop checkpoints.

### Option 3: Individual Steps

Run each step manually:

```bash
# Step 1: Deep Topology Analysis
python topology_analyzer.py "Mar Vista POC 100%_CheckPrint_20211118 Complete.pdf"

# Step 2: Generate Test Scenarios
python scenario_generator.py

# Step 3: Run Test Simulation
python test_simulator.py

# Step 4: View Dashboard
start training_dashboard.html  # Or open in browser
```

## 📊 Training Pipeline Steps

### Step 1: Deep Topology Analysis

**Purpose:** Comprehensively extract structural elements, relationships, and specifications from the Mar Vista project.

**What it does:**
- Analyzes every page using GPT-4 for text understanding
- Extracts detailed topology: elements, materials, dimensions, connections
- Identifies relationships between structural components
- Creates `enhanced_topology.json` with complete project topology

**Output:**
```json
{
  "project_name": "Mar Vista POC...",
  "total_pages": 42,
  "topology": {
    "structural_elements": {...},
    "foundation_systems": {...},
    "connection_types": {...},
    "materials": {...},
    "dimensions": {...}
  },
  "relationships": [...],
  "summary": {...}
}
```

**Estimated time:** 10-15 minutes  
**Estimated cost:** $2-4 (OpenAI API)

**Human checkpoint:** Review topology accuracy before proceeding

### Step 2: Scenario Generation

**Purpose:** Create comprehensive search test scenarios covering 90% of the knowledge base.

**Scenario types generated:**
- **Direct element searches** - "CIDH pile", "steel shear key"
- **Compound queries** - "bearing pad abutment", "pipe pin connection"
- **Specification searches** - Material grades, codes, standards
- **Dimensional queries** - "girder spacing", "pile depth"
- **Relationship queries** - "column supports", "foundation system"
- **Terminology variations** - Synonyms and technical terms

**Output:** `test_scenarios.json` with 80-150 test scenarios

**Estimated time:** 1-2 minutes  
**No API cost** (local processing)

**Human checkpoint:** Review scenarios for completeness

### Step 3: Test Simulation

**Purpose:** Execute all scenarios, validate results, generate improvement feedback.

**What it does:**
- Runs each scenario against the search API
- Compares actual results with expected ground truth
- Calculates precision, recall, F1 scores
- Identifies patterns in failures
- Generates actionable improvement feedback
- Creates detailed test report

**Metrics calculated:**
- **Precision:** Are the returned results relevant?
- **Recall:** Did we find all relevant pages?
- **Pass Rate:** How many scenarios passed validation?
- **Response Time:** Average search speed

**Output:** `test_results_YYYYMMDD_HHMMSS.json`

**Estimated time:** 5-10 minutes (depends on scenario count)  
**No API cost** (queries existing RAG system)

**Human checkpoint:** Review failures and feedback

### Step 4: Training Dashboard

**Purpose:** Visual interface to review training results and feedback.

**Features:**
- **Pass rate metrics** with visual progress bars
- **Precision/Recall charts**
- **Feedback recommendations** categorized by severity
- **Detailed results table** with filters
- **Scenario performance breakdown**

**Usage:**
1. Open `training_dashboard.html` in browser
2. Load the latest `test_results_*.json` file
3. Review metrics and identify improvement areas
4. Apply feedback recommendations

**Human checkpoint:** Decide which improvements to implement

## 🔄 Feedback Loop & Self-Improvement

The simulator automatically detects issues and generates feedback:

### High Severity Issues
- **Pattern failures:** Multiple failures in same scenario type
- **Low recall:** Missing many expected pages
- Recommendation: Improve chunking, increase k parameter, adjust threshold

### Medium Severity Issues
- **Low precision:** Returning too many irrelevant results
- **Missing pages:** Specific pages frequently missed
- Recommendation: Review vision analysis, improve topology extraction

### Self-Correction Process

1. **Review feedback** in the dashboard
2. **Apply fixes:**
   - Update `topology_analyzer.py` to extract missed elements
   - Adjust `config.py` parameters (CHUNK_SIZE, CHUNK_OVERLAP)
   - Modify `vision_analyzer.py` prompts for better extraction
   - Update `api.py` relevance thresholds
3. **Re-run affected steps:**
   ```bash
   # If topology needs improvement
   python topology_analyzer.py "Mar Vista POC 100%_CheckPrint_20211118 Complete.pdf"
   
   # Regenerate scenarios
   python scenario_generator.py
   
   # Re-test
   python test_simulator.py
   ```
4. **Compare results** - Load both test results in dashboard to see improvement

## 📈 Success Metrics

### Target Goals
- ✅ **Pass Rate:** >85%
- ✅ **Average Precision:** >80%
- ✅ **Average Recall:** >85%
- ✅ **Response Time:** <2 seconds
- ✅ **Coverage:** 90% of knowledge base elements

### Interpreting Results

**Excellent Performance:**
- Pass rate: 90%+
- Precision & Recall: 85%+
- Few or no feedback items

**Good Performance:**
- Pass rate: 75-90%
- Precision & Recall: 70-85%
- Minor improvements needed

**Needs Improvement:**
- Pass rate: <75%
- Precision or Recall: <70%
- Multiple high-severity feedback items

## 🛠️ Troubleshooting

### "API server not available"
**Solution:** Start the API server first:
```bash
python api.py
```

### "Topology file not found"
**Solution:** Run Step 1 first:
```bash
python topology_analyzer.py "Mar Vista POC 100%_CheckPrint_20211118 Complete.pdf"
```

### "Low recall across many scenarios"
**Possible causes:**
- Relevance threshold too strict
- Chunking strategy missing content
- k parameter too small

**Solutions:**
- Increase `relevance_threshold` in test_simulator.py
- Adjust CHUNK_SIZE/CHUNK_OVERLAP in config.py
- Increase k parameter in searches

### "Low precision across many scenarios"
**Possible causes:**
- Relevance threshold too permissive
- Topology extraction including noise

**Solutions:**
- Decrease `relevance_threshold`
- Improve topology extraction prompts
- Refine search query understanding

## 💡 Advanced Usage

### Custom Scenario Generation

Edit `scenario_generator.py` to add custom scenario types:

```python
def _generate_custom_scenarios(self):
    """Add your custom scenario logic here."""
    # Example: Generate scenarios for specific detail types
    pass
```

### Adjusting Coverage Target

In `scenario_generator.py`, change the coverage target:

```python
self.coverage_target = 0.95  # 95% coverage instead of 90%
```

### Parallel Testing

For faster testing, modify test_simulator.py to use async:

```python
import asyncio
import aiohttp

# Implement parallel scenario execution
```

## 📁 File Structure

```
training_controller.py       # Master controller for pipeline
topology_analyzer.py         # Step 1: Deep topology extraction
scenario_generator.py        # Step 2: Test scenario generation
test_simulator.py           # Step 3: Simulation & validation
training_dashboard.html     # Step 4: Visual dashboard
enhanced_topology.json      # Output: Enhanced topology
test_scenarios.json         # Output: Generated scenarios
test_results_*.json         # Output: Test results
```

## 🔒 Privacy & Costs

### API Costs
- **Step 1 (Topology):** $2-4 for Mar Vista (42 pages)
- **Step 2 (Scenarios):** Free (local processing)
- **Step 3 (Testing):** Free (uses existing RAG system)
- **Step 4 (Dashboard):** Free (local HTML)

**Total:** ~$2-4 per full training run

### Data Privacy
- All processing happens locally except OpenAI API calls
- No data sent to third parties
- Training results stored locally

## 📞 Support & Questions

For issues or questions:
1. Check the troubleshooting section above
2. Review test_results_*.json for detailed error messages
3. Check feedback items in dashboard for recommendations

## 🔄 Next Steps

After completing the training program:

1. **Review feedback** and implement high-priority improvements
2. **Re-index Mar Vista** if topology was enhanced:
   ```bash
   python main.py update "Mar Vista POC 100%_CheckPrint_20211118 Complete.pdf"
   ```
3. **Expand to other projects** - Run training on additional PDFs
4. **Iterate** - Run training periodically to maintain quality
5. **Monitor** - Track metrics over time to ensure improvements

## ✅ Checklist

Before starting:
- [ ] API server is running (`python api.py`)
- [ ] Mar Vista project is indexed in knowledge base
- [ ] OpenAI API key is configured
- [ ] At least 30 minutes available for full pipeline

Ready to start? Run:
```bash
python training_controller.py
```
