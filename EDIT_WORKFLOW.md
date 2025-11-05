# SQA Edit Workflow - Mirrored Architecture

## Overview

The SQA Edit Workflow allows users to modify existing research reports based on natural language instructions. The workflow **mirrors the existing 4-step report generation pipeline** with edit-specific extensions to handle current report context.

## Key Design Principle: Mirroring Existing Flow

The edit workflow is designed as a **parallel mirror** of the existing `run_qa_pipeline`:

| Step | Original Pipeline | Edit Pipeline | Key Difference |
|------|-------------------|---------------|----------------|
| **0** | (none) | Retrieve current report | EDIT-SPECIFIC: Fetch existing report from thread_id |
| **1** | Query decomposition → Search → Rerank | Conditional search + mentioned_papers → Rerank | Adds mentioned_papers, conditional search |
| **2** | Quote extraction (`step_select_quotes`) | Quote extraction (`step_select_quotes_edit`) | Extended prompts include current report context |
| **3** | Clustering/Planning (`step_clustering`) | Edit planning (`step_clustering_edit`) | Extended prompts include current sections + edit actions |
| **4** | Section generation (`generate_iterative_summary`) | Section editing (`generate_iterative_summary_edit`) | Extended prompts include current section content + actions |
| **5** | Post-processing (citations, tables) | Post-processing (citations, tables) | SAME as original |

## Architecture Details

### Extended Prompts (Not New Prompts!)

All prompts in `llms/edit_prompts.py` are **extensions** of the original prompts in `llms/prompts.py`:

1. **`SYSTEM_PROMPT_QUOTE_PER_PAPER_EDIT`** (extends `SYSTEM_PROMPT_QUOTE_PER_PAPER`)
   - **Added inputs**: current_sections_summary, edit_instruction
   - **Same logic**: Extract exact quotes from papers
   - **New context**: Guides LLM to focus on quotes relevant to edit instruction

2. **`SYSTEM_PROMPT_QUOTE_CLUSTER_EDIT`** (extends `SYSTEM_PROMPT_QUOTE_CLUSTER`)
   - **Added inputs**: current_report, edit_instruction
   - **Same logic**: Cluster quotes into dimensions
   - **New context**: Outputs include "action" field (KEEP, EXPAND, ADD_TO, REPLACE, DELETE, NEW)

3. **`PROMPT_ASSEMBLE_SUMMARY_EDIT`** (extends `PROMPT_ASSEMBLE_SUMMARY`)
   - **Added inputs**: current_section_content, action, edit_instruction
   - **Same logic**: Generate section with citations
   - **New context**: Handles different actions (keep existing, expand, replace, etc.)

### Mirrored Pipeline Structure

The `EditPipeline` class (in `rag/edit_pipeline.py`) mirrors `MultiStepQAPipeline`:

```python
# Original: MultiStepQAPipeline
class MultiStepQAPipeline:
    def step_select_quotes(query, scored_df, sys_prompt)
    def step_clustering(query, per_paper_summaries, sys_prompt)
    def generate_iterative_summary(query, per_paper_summaries_extd, plan, sys_prompt)

# Edit version: EditPipeline (SAME structure, extended inputs)
class EditPipeline:
    def step_select_quotes_edit(edit_instruction, current_report, scored_df)
    def step_clustering_edit(edit_instruction, current_report, per_paper_summaries)
    def generate_iterative_summary_edit(edit_instruction, current_report, per_paper_summaries_extd, plan)
```

### Data Flow Comparison

**Original Pipeline:**
```
query string
  ↓
[Step 1] find_relevant_papers → rerank → scored_df
  ↓
[Step 2] step_select_quotes → per_paper_summaries (quotes)
  ↓
[Step 3] step_clustering → ClusterPlan (dimensions with quote indices)
  ↓
[Step 4] extract_quote_citations → per_paper_summaries_extd (with inline citations)
  ↓
[Step 5] generate_iterative_summary → sections (one at a time)
  ↓
TaskResult (new report)
```

**Edit Pipeline:**
```
edit_instruction + thread_id + mentioned_papers
  ↓
[Step 0] retrieve current report
  ↓
[Step 1] mentioned_papers → candidates
         optional search (if needed) → candidates
         rerank → scored_df
  ↓
[Step 2] step_select_quotes_EDIT → per_paper_summaries
         (WITH current report context)
  ↓
[Step 3] step_clustering_EDIT → EditClusterPlan
         (WITH current report, outputs actions: KEEP/EXPAND/ADD_TO/REPLACE/DELETE/NEW)
  ↓
[Step 4] extract_quote_citations → per_paper_summaries_extd
         (SAME as original)
  ↓
[Step 5] generate_iterative_summary_EDIT → sections
         (WITH current section content + action for each)
  ↓
TaskResult (edited report)
```

## mentioned_papers (paper_ids) Usage

The `mentioned_papers` parameter (list of corpus_ids) is used at **three critical points** in the pipeline:

### MARKER #1: Fetching Mentioned Papers
**Location**: `run_edit_pipeline`, lines ~120-140
**Current implementation**:
```python
if req.mentioned_papers:
    mentioned_metadata = get_paper_metadata(req.mentioned_papers)
    # Fetch basic metadata: title, abstract, authors, year, citations
```

**What we WOULD do if we had access**:
- Fetch full paper text or specific passages
- Get rich snippet-level content (like semantic search returns)
- Extract section-level content from papers

**Why it matters**: Currently we only use abstracts for mentioned papers, while searched papers come with rich snippets. This creates an asymmetry.

### MARKER #2: Converting to Search Format
**Location**: `run_edit_pipeline`, lines ~140-165
**Current implementation**:
```python
for corpus_id in req.mentioned_papers:
    candidate = {
        "text": mdata.get("abstract", ""),  # ← Only abstract available!
        "score": 1.0,  # High score - user explicitly mentioned
    }
    retrieved_candidates.append(candidate)
```

**What we WOULD do if we had access**:
- Use actual paper snippets instead of full abstract
- Include passage-level information with offsets
- Maintain same granularity as search results

### MARKER #3: Quote Extraction Context
**Location**: `edit_pipeline.py`, `step_select_quotes_edit`
**Current behavior**:
- LLM extracts quotes from abstract (for mentioned papers)
- LLM extracts quotes from rich snippets (for searched papers)

**What we WOULD do if we had access**:
- Extract from same granular content for both
- Include section context and PDF locations
- Maintain consistency in quote quality

## API Usage

### Input Parameters

```python
{
    "edit_existing": true,              # Flag to trigger edit workflow
    "thread_id": "task-id-123",         # ID of thread containing current report
    "edit_instruction": "Add these papers to section 2",  # Decontextualized instruction
    "mentioned_papers": [12345, 67890], # Optional: corpus_ids of specific papers
    "query": "Optional context"         # Optional: original query for context
}
```

### Edit Actions

The edit plan generates one of these actions for each section:

| Action | Meaning | Example Instruction |
|--------|---------|-------------------|
| `KEEP` | No changes needed | (implicit for unaffected sections) |
| `EXPAND` | Add more content | "Expand section 3 with more detail" |
| `ADD_TO` | Add specific papers | "Add these papers to section 2" |
| `REPLACE` | Replace content | "Replace intro with background on X" |
| `DELETE` | Remove section | "Remove the limitations section" |
| `NEW` | Create new section | "Add a section about Y" |

### Example Requests

#### 1. Add user-mentioned papers (uses mentioned_papers)
```json
{
    "edit_existing": true,
    "thread_id": "abc-123",
    "edit_instruction": "Add these papers to the report",
    "mentioned_papers": [123456789, 987654321]
}
```
**Flow**: mentioned_papers → fetch metadata → add to scored_df → quote extraction → add to relevant sections

#### 2. Add papers by topic (uses search)
```json
{
    "edit_existing": true,
    "thread_id": "abc-123",
    "edit_instruction": "Add more papers about transformer architectures"
}
```
**Flow**: edit_instruction triggers search → rerank → quote extraction → add to relevant sections

#### 3. Combination (mentioned_papers + search)
```json
{
    "edit_existing": true,
    "thread_id": "abc-123",
    "edit_instruction": "Add this paper and others like it about attention mechanisms",
    "mentioned_papers": [123456789]
}
```
**Flow**: mentioned_papers + search → combined scored_df → quote extraction → add to sections

## Implementation Files

### New Files

1. **`api/scholarqa/llms/edit_prompts.py`** (~300 lines)
   - Extended versions of original prompts
   - Clearly marked as extensions, not replacements
   - Same structure, additional input placeholders

2. **`api/scholarqa/rag/edit_pipeline.py`** (~368 lines)
   - `EditPipeline` class mirroring `MultiStepQAPipeline`
   - Three main methods: `step_select_quotes_edit`, `step_clustering_edit`, `generate_iterative_summary_edit`
   - Each method has same signature pattern as original + edit context

### Modified Files

1. **`api/scholarqa/models.py`**
   - Extended `ToolRequest` with: `edit_existing`, `thread_id`, `edit_instruction`, `mentioned_papers`

2. **`api/scholarqa/scholar_qa.py`**
   - Added `run_edit_pipeline()` method (~280 lines)
   - Mirrors `run_qa_pipeline()` structure exactly
   - Added `_retrieve_report_from_thread()` helper
   - Initialized `EditPipeline` in constructor

3. **`api/scholarqa/app.py`**
   - Added routing: checks `edit_existing` flag
   - Routes to `run_edit_pipeline()` if true, `run_qa_pipeline()` if false

## Reused Components

The edit workflow **maximizes reuse** of existing components:

- ✅ `find_relevant_papers()` - SAME
- ✅ `rerank_and_aggregate()` - SAME
- ✅ `extract_quote_citations()` - SAME
- ✅ `get_json_summary()` - SAME
- ✅ `get_paper_metadata()` - SAME
- ✅ `gen_table_thread()` - SAME
- ✅ `EventTrace` - SAME
- ✅ Cost tracking (`CostAwareLLMCaller`) - SAME
- ✅ State management (`state_mgr`) - SAME

Only 3 NEW methods in `EditPipeline`:
1. `step_select_quotes_edit` (extended version)
2. `step_clustering_edit` (extended version)
3. `generate_iterative_summary_edit` (extended version)

## Test Cases

### Basic Operations
1. ✅ Add single user-mentioned paper
2. ✅ Add multiple user-mentioned papers (10+)
3. ✅ Search and add papers by topic
4. ✅ Add recent papers (with date filter)
5. ✅ Add more papers (general expansion)

### mentioned_papers Specific
6. ✅ Add paper by ID only
7. ✅ Add paper by ID with context ("add this paper to section 2")
8. ✅ Add papers like a mentioned paper ("add papers similar to [ID]")

### Section Operations
9. ✅ Expand specific section
10. ✅ Remove and replace section
11. ✅ Fix errors in section
12. ✅ Create new section

## Future Improvements

### mentioned_papers Enhancement
**Priority**: HIGH
**Impact**: Improves quality for user-specified papers

Currently mentioned_papers only uses abstracts. To improve:
1. Fetch full paper text if available
2. Extract relevant passages automatically
3. Use same snippet-based approach as search
4. Maintain passage-to-PDF mapping

### Other Enhancements
- Interactive plan approval before execution
- Diff view showing changes
- Undo/redo support
- Batch edit instructions
- Section-specific targeting ("edit section 2")

## Summary

The edit workflow is a **true mirror** of the existing pipeline:
- Same 4-step structure (+ step 0 for retrieval)
- Same data flow (scored_df → quotes → plan → sections)
- Same post-processing (citations, tables)
- **Extended prompts** that add edit context to existing prompts
- **Extended inputs** (current_report, edit_instruction, mentioned_papers)
- **Extended outputs** (actions for each section)

This design ensures:
- Consistency with existing behavior
- Easy maintenance (prompts evolve together)
- Reuse of battle-tested components
- Clear extension points for future improvements
