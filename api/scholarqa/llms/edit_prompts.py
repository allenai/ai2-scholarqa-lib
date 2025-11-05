"""
Prompts for the edit workflow steps.
"""

# Step 1a: Decide if new search is needed
PROMPT_DECIDE_SEARCH = """
You are helping to edit an existing research report based on a user's instruction.

Your task is to decide whether we need to search for NEW papers to fulfill the edit instruction,
or whether we can work with the papers already in the report and any papers explicitly mentioned by the user.

<current_report>
{current_report}
</current_report>

<edit_instruction>
{edit_instruction}
</edit_instruction>

<mentioned_papers>
{mentioned_papers}
</mentioned_papers>

Think step by step:
1. What is the user asking for?
2. Do we already have sufficient papers in the current report to address this?
3. Did the user mention specific papers to add?
4. Do we need to find additional papers?

If we need new papers, formulate a clear search query that will help find relevant papers.

Your response should be in JSON format following the SearchDecision schema.
"""

# Step 2: Generate per-section edit plan
PROMPT_GENERATE_EDIT_PLAN = """
You are helping to edit an existing research report based on a user's instruction.

Your task is to create a detailed plan for how to edit each section of the report.

<current_report>
Title: {report_title}

Sections:
{sections_summary}
</current_report>

<edit_instruction>
{edit_instruction}
</edit_instruction>

<mentioned_papers>
{mentioned_papers}
</mentioned_papers>

<available_papers>
{available_papers}
</available_papers>

Think step by step:
1. What is the user asking for?
2. Which sections are affected by this instruction?
3. For each section, what action should we take?

Available actions for each section:
- "keep": Leave the section unchanged
- "expand": Expand the section with more detail (e.g., "go deeper on this topic")
- "add_papers": Add specific papers to the section
- "delete": Remove the section entirely
- "go_deeper": Go deeper on the topic with more comprehensive analysis
- "replace": Replace the section with entirely new content
- "modify": Modify the section based on a specific instruction

For "go deeper" actions:
- This means providing more comprehensive analysis and detail
- Include more nuanced discussion of methods, findings, and implications
- Add more papers if available
- Provide deeper insights into the topic

Create a plan for EVERY existing section. Even if a section won't change, include it with action "keep".

If the instruction requires adding new sections (e.g., "add a section about X"), include those in the "new_sections" field.

Your response should be in JSON format following the EditPlan schema.
"""

# Step 3: Execute edits for a specific section
PROMPT_EXECUTE_SECTION_EDIT = """
You are editing a section of a research report based on a specific action plan.

<original_section>
Title: {section_title}
Content:
{section_content}
</original_section>

<edit_action>
{edit_action}
</edit_action>

<edit_instruction>
{specific_instruction}
</edit_instruction>

<full_report_context>
{full_report_context}
</full_report_context>

<available_references>
{section_references}
</available_references>

Instructions based on action type:

For "expand" or "go_deeper":
- Keep all existing content and citations
- Add new paragraphs that go into more depth
- Use additional references if provided
- For "go_deeper" specifically: provide comprehensive analysis with more nuanced discussion of methods, findings, implications, and connections between papers

For "add_papers":
- Integrate the new papers naturally into the existing narrative
- Add citations in the format [corpus_id | Author, Year | Citations: count]
- Ensure the new content flows well with existing content

For "modify":
- Make the specific changes requested in the instruction
- Preserve citations unless specifically asked to remove them
- Maintain the overall structure unless instructed otherwise

For "replace":
- Write entirely new content based on the instruction
- Use the provided references
- Follow the same citation format

For "keep":
- Return the original content unchanged

Citation format:
- Use inline citations like [corpus_id | Author, Year | Citations: count]
- Multiple citations: [corpus_id1 | Author1, Year1 | Citations: count1][corpus_id2 | Author2, Year2 | Citations: count2]
- Citations should be based on the content from the references provided

Write the edited section below. Output ONLY the edited section text with citations, no additional commentary.
"""

# Prompt for creating a brand new section
PROMPT_CREATE_NEW_SECTION = """
You are creating a new section for a research report based on a user's instruction.

<full_report>
{full_report}
</full_report>

<new_section_instruction>
Title: {section_title}
Instruction: {section_instruction}
</new_section_instruction>

<available_references>
{section_references}
</available_references>

Create a new section that:
1. Fits naturally with the existing report structure
2. Addresses the instruction clearly
3. Uses the provided references with proper citations
4. Maintains the same writing style as the rest of the report

Citation format:
- Use inline citations like [corpus_id | Author, Year | Citations: count]
- Multiple citations: [corpus_id1 | Author1, Year1 | Citations: count1][corpus_id2 | Author2, Year2 | Citations: count2]

Write the new section below. Output ONLY the section text with citations, no additional commentary.
"""

# User prompt formats for the edit workflow
USER_PROMPT_SEARCH_DECISION_FORMAT = """
Here is the information for the search decision task:

{formatted_input}
"""

USER_PROMPT_EDIT_PLAN_FORMAT = """
Here is the information for the edit plan generation task:

{formatted_input}
"""
