# SQA Edit Workflow

## Overview

The SQA Edit Workflow allows users to modify existing research reports based on natural language instructions. Instead of generating a new report from scratch, the workflow intelligently edits the current report by adding papers, expanding sections, removing content, or making other targeted modifications.

## Architecture

The edit workflow consists of three main steps:

### Step 1a: Decide Search Strategy
Analyzes the edit instruction and current report to determine if searching for new papers is necessary.

**Output**: `SearchDecision` containing:
- `needs_search`: Boolean indicating if a search is required
- `search_query`: The query to use if search is needed
- `reasoning`: Explanation for the decision

### Step 1b: Conditional Search (if needed)
If Step 1a determines that new papers are needed, this step:
- Performs semantic search on the corpus
- Reranks results using cross-encoder
- Extracts paper metadata and content

### Step 2: Generate Edit Plan
Creates a detailed plan for how to edit each section of the report.

**Output**: `EditPlan` containing:
- `section_plans`: List of `SectionEditPlan` for each existing section
- `new_sections`: List of new sections to add

**Available Edit Actions**:
- `keep`: Leave section unchanged
- `expand`: Add more detail to the section
- `add_papers`: Integrate specific papers into the section
- `delete`: Remove the section entirely
- `go_deeper`: Provide comprehensive analysis with more depth
- `replace`: Replace section with entirely new content
- `modify`: Make specific changes based on instruction

### Step 3: Execute Edit Plan
Executes the edit plan section by section:
- Edits existing sections according to their action plans
- Creates new sections if specified
- Updates citations and TLDRs
- Maintains consistent formatting

## API Usage

### Input Parameters

The edit workflow uses the existing `/query_corpusqa` endpoint with additional parameters:

```python
{
    "edit_existing": true,              # Flag to trigger edit workflow
    "thread_id": "task-id-123",         # ID of the thread containing the report
    "edit_instruction": "Add these papers to section 2",  # Natural language instruction
    "mentioned_papers": [12345, 67890], # Optional: corpus_ids mentioned by user
    "query": "Original query context"   # Optional: for context
}
```

### Example Requests

#### 1. Add specific papers to the report
```json
{
    "edit_existing": true,
    "thread_id": "abc-123",
    "edit_instruction": "Add these papers to the report",
    "mentioned_papers": [123456789, 987654321]
}
```

#### 2. Go deeper on a specific section
```json
{
    "edit_existing": true,
    "thread_id": "abc-123",
    "edit_instruction": "Go deeper on section 3 with more comprehensive analysis of the methods and their implications"
}
```

#### 3. Add papers matching a topic
```json
{
    "edit_existing": true,
    "thread_id": "abc-123",
    "edit_instruction": "Add more recent papers about transformer architectures to section 2"
}
```

#### 4. Remove and replace content
```json
{
    "edit_existing": true,
    "thread_id": "abc-123",
    "edit_instruction": "Remove section 4 and replace it with a discussion of limitations"
}
```

#### 5. Fix specific issues
```json
{
    "edit_existing": true,
    "thread_id": "abc-123",
    "edit_instruction": "Fix the citations in section 2 and update the methodology description"
}
```

## Test Cases

The following test cases are useful for validating the edit workflow:

### Basic Operations
1. **Add single paper**: "Add this paper to the report" + `mentioned_papers: [corpus_id]`
2. **Add multiple papers**: "Add these 10 papers to the report" + `mentioned_papers: [ids]`
3. **Search and add**: "Add papers to the report about blah" (triggers search)
4. **Add more papers**: "Add MORE papers to the report" (triggers search)
5. **Add recent papers**: "Add more recent papers" (triggers filtered search)

### Section-Specific Operations
6. **Remove and replace**: "Remove this section and replace with blah"
7. **Add similar papers**: "Add this paper and papers like it" + `mentioned_papers: [id]`
8. **Fix errors**: "Fix THESE errors in section 2"
9. **Go deeper**: "Go deeper on section 5" or "Provide more detail about X in section 3"

### Advanced Operations
10. **Extract section**: "Make another report about only what's in section 2"
11. **Expand with topic**: "Expand section 4 to include discussion of X topic"
12. **Shorten section**: "Shorten section 3 to focus only on key findings"

## Implementation Details

### File Structure

New files created for the edit workflow:

```
api/scholarqa/
├── models_edit.py              # Data models for edit workflow
├── llms/edit_prompts.py        # Prompts for edit steps
└── rag/edit_pipeline.py        # Edit pipeline implementation
```

Modified files:

```
api/scholarqa/
├── models.py                   # Extended ToolRequest with edit fields
├── scholar_qa.py              # Added run_edit_pipeline() method
└── app.py                     # Added routing logic for edit requests
```

### Reusing Existing Components

The edit workflow maximizes reuse of existing pipeline components:

- **Retrieval & Reranking**: Uses existing `find_relevant_papers()` and `rerank_and_aggregate()`
- **Citation Extraction**: Uses existing `get_json_summary()` for post-processing
- **LLM Calling**: Uses existing `llm_completion()` and cost tracking
- **State Management**: Uses existing state manager for report storage/retrieval
- **Event Tracing**: Uses existing `EventTrace` for monitoring and debugging

### Prompts

All prompts for the edit workflow are in `llms/edit_prompts.py`:

- `PROMPT_DECIDE_SEARCH`: For Step 1a (search decision)
- `PROMPT_GENERATE_EDIT_PLAN`: For Step 2 (edit planning)
- `PROMPT_EXECUTE_SECTION_EDIT`: For Step 3 (section editing)
- `PROMPT_CREATE_NEW_SECTION`: For creating new sections

Each prompt includes:
- Clear task description
- Available actions and their meanings
- Examples and guidelines
- "Go deeper" specific instructions for comprehensive analysis

## "Go Deeper" Feature

The "go deeper" action provides more comprehensive analysis by:

- Including more nuanced discussion of methods and findings
- Adding detailed implications and connections between papers
- Providing deeper insights into the topic
- Incorporating additional papers if available
- Expanding on theoretical frameworks and practical applications

This addresses the user feedback about wanting follow-up reports with more depth.

## Output

The edit workflow returns a standard `TaskResult` containing:

- `report_title`: Original or updated title
- `sections`: List of edited/new sections
- `cost`: Total cost of the edit operation
- `tokens`: Token usage breakdown

Additionally, an `EditResult` summary is logged containing:

- `summary`: High-level description of changes
- `sections_modified`: Indices of modified sections
- `sections_added`: Indices of new sections
- `sections_deleted`: Indices of deleted sections
- `papers_added`: Corpus IDs of newly added papers

## Error Handling

The workflow validates:

- `edit_existing` flag is true
- `thread_id` is provided
- `edit_instruction` is provided
- Report exists for the given `thread_id`
- Section indices in edit plan are valid

If validation fails, appropriate error messages are returned.

## Future Enhancements

Potential improvements to the edit workflow:

1. **Interactive planning**: Show edit plan to user for approval before execution
2. **Undo/redo**: Track edit history for rollback
3. **Diff view**: Show changes between original and edited versions
4. **Batch edits**: Support multiple edit instructions in one request
5. **Smart merging**: Better handling of conflicting edits
6. **Section targeting**: Direct section references like "section 2" or "Introduction"
7. **Quality metrics**: Assess edit quality and coherence

## Examples

See the test cases above for comprehensive examples of each edit operation.
