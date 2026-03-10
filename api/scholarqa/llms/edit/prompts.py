# ============================================================================
# EDIT WORKFLOW PROMPTS
# ============================================================================

EDIT_INTENT_ANALYZER_PROMPT = """
<task>
Analyze this edit instruction for an existing research report.

Determine:
1. **Search Query**: If new papers need to be added to the report, compose a short search query from the original report context with the edit intent. The query should be meaningful to search over semantically over an index, not just a combination of keywords. Leave EMPTY if no search is needed.

2. **Search Constraints**: Extract any filters (year, venue, author, citations) to apply to the search.

3. **Papers to Add**: List corpus_ids that user explicitly wants added (from mentioned papers).

4. **Papers to Remove**: Look at the current_citations below and identify which corpus_ids should be removed based on the instruction. If the instruction mentions a constraint eg. "remove papers before 2020", look at the citations and output the actual corpus_ids of papers with year < 2020.

5. **Stylistic Edit**: Is this purely a style/structure change with no paper modifications?

6. **Section Targeting**: Which sections are affected? All or specific ones?

The current year is 2026. Interpret "recent" as 2023-2026.
</task>

<current_report>
Original Query: {original_query}

{report_context}
</current_report>

<current_citations>
These are all papers currently cited in the report. Use this to resolve constraint-based removal (e.g., "only keep highly cited papers" → find corpus_ids with citations < 50):
{current_citations}
</current_citations>

<edit_instruction>
{edit_instruction}
</edit_instruction>

<mentioned_papers>
User explicitly mentioned these papers (corpus_ids): {corpus_ids}
</mentioned_papers>

<mentioned_sections>
User mentioned these section titles: {section_titles}
</mentioned_sections>

<decision_rules>
**search_query should be NON-EMPTY when:**
- "Add papers about X" → search_query = rephrase(original_query + X topic)
- "Add more papers" → search_query = rephrase(original_query + all current section topic)
- "Add papers to section S about X" → search_query = rephrase(original_query + X + section topic for S)
- "Add recent papers on Y" → search_query = rephrase(original_query + Y, earliest_year = 2022)
- "Add papers from venue Z" → search_query = rephrase(original_query + all current section topics, venues = Z)
- "Add paper 12345" (specific corpus_ids provided) → search_query = rephrase(original_query + all current section topics), papers_to_add = [12345]

**search_query should be EMPTY when:**
- "Remove paper 12345" → papers_to_remove = [12345]
- "Remove papers from venue X" → Look at current_citations, find papers with that venue, output their corpus_ids in papers_to_remove
- "Remove papers before 2020" → Look at current_citations, find papers with year < 2020, output their corpus_ids in papers_to_remove
- "Rewrite section 2" / "Fix errors" / "Shorten" → is_stylistic = true

**Composing search_query:**
When search is needed, combine:
- Original query context (domain/topic)
- New topic from edit instruction
- Target section's topic if specific section mentioned else topics from all sections
The search query should be a natural language phrase suitable for semantic search, not just a collection of keywords.
Also, do not include any constraints (year, venue, author) in the search_query itself; those go in separate fields.
Example: Original="AI safety", Edit="Add papers about robustness to Methods"
→ search_query = "AI safety methods related to robustness testing"

**Resolving removal constraints:**
When the instruction asks to remove papers by constraint (year, venue, author, citations):
1. Look at the current_citations JSON
2. Filter papers matching the constraint
3. Output their corpus_ids in papers_to_remove
Example: "Remove papers older than 2020" + current_citations has papers from 2018, 2019, 2021
→ papers_to_remove = ["corpus_id_of_2018_paper", "corpus_id_of_2019_paper"]
</decision_rules>

<examples>
Example 1:
Original Query: "AI safety methods"
Edit Instruction: "Add papers about robustness testing"
Output:
{{
  "cot": "User wants to add papers about robustness. Need to search combining original query with new topic.",
  "search_query": "AI safety methods related to robustness testing",
  "is_stylistic": false,
  "affects_all_sections": true
}}

Example 2:
Edit Instruction: "Remove paper 12345 and paper 67890"
Output:
{{
  "cot": "User wants to remove specific papers by ID. No search needed.",
  "search_query": "",
  "is_stylistic": false,
  "papers_to_remove": ["12345", "67890"],
  "affects_all_sections": true
}}

Example 3:
Edit Instruction: "Rewrite section Methods to be more concise"
Output:
{{
  "cot": "Stylistic change to rewrite a section. No paper changes.",
  "search_query": "",
  "is_stylistic": true,
  "target_sections": ["Methods"],
  "affects_all_sections": false
}}

Example 4:
Edit Instruction: "Remove papers from before 2020"
current_citations includes: [{{"corpus_id": "111", "year": 2018}}, {{"corpus_id": "222", "year": 2021}}, {{"corpus_id": "333", "year": 2019}}]
Output:
{{
  "cot": "User wants to remove papers by year constraint. Looking at current_citations, papers 111 (2018) and 333 (2019) are before 2020.",
  "search_query": "",
  "is_stylistic": false,
  "papers_to_remove": ["111", "333"],
  "affects_all_sections": true
}}

Example 5:
Edit Instruction: "Add papers 123 and 456 to the report"
mentioned_papers: [123, 456]
Output:
{{
  "cot": "User provided specific corpus_ids. No search needed, just add these papers.",
  "search_query": "",
  "is_stylistic": false,
  "papers_to_add": ["123", "456"],
  "affects_all_sections": true
}}

Example 6:
Original Query: "Training techniques for large language models"
Edit Instruction: "Add recent papers on reinforcement learning to the Methods section"
Output:
{{
    "cot": "User wants to add recent papers on reinforcement learning specifically to the Methods section
. Need to search combining original query with new topic and section i.e. training methods like reinforcement learning",
    "search_query": "LLM training methods related to reinforcement learning",
    "earliest_search_year": "2023",
    "is_stylistic": false,
    "affects_all_sections": false,
    "target_sections": ["Methods"]
}}

Example 7:
Edit Instruction: "Remove papers by John Doe"
Output:
{{
  "cot": "User wants to remove papers by author John Doe. Ppaers by John Doe in current_citations are corpus_id_1 and corpus_id_2.",
  "search_query": "",
  "is_stylistic": false,
  "papers_to_remove": ["corpus_id_1", "corpus_id_2],
    "affects_all_sections": true
}}

Example 8:
Edit Instruction: "Add papers by John Doe from NeurIPS 2022"
Output:
{{
  "cot": "User wants to add papers by John Doe from NeurIPS 2022. Need to search combining original query with all section titles without the added constraints.",
  "search_query": "Techniques related to training large language models, specifically <section topics>",
  "earliest_search_year": "2022", "latest_search_year": "2022",
  "venues": ["NeurIPS"],
  "authors": ["John Doe"],

</examples>


Output a JSON object following the EditIntentAnalysis schema. ONLY output the fields that are non-empty or relevant to the instruction, pay special attention to the list types.
"""

# ============================================================================
# EDIT STEP 1: QUOTE EXTRACTION (mirrors SYSTEM_PROMPT_QUOTE_PER_PAPER)
# ============================================================================

SYSTEM_PROMPT_QUOTE_PER_PAPER_EDIT = """
In this task, you are presented with an existing report and an academic paper with snippets and metadata.

The user wants to update their existing report with new papers. Your job is to extract relevant quotes from this paper.

Here is the original query that generated the report:<original_query>
<original_query>
{original_query}
</original_query>

Here is the search query that returned the paper below:
<search_query>
{search_query}
</search_query>

<current_report>
{report_context}
</current_report>

Stitch together text from the paper content that is relevant to the query and report.

To be clear, copy EXACT text ONLY.

Include any references that are part of the text to be copied. The references can occur at the beginning, middle, or end of the text.

eg, if you chose to include the text "(Moe et al., 2020) show that A is very important for B (Miles, 2023) and this has been known since 2024 [1][2]",
it's critical that all the references (Moe et al., 2020), (Miles, 2023), [1] and [2] are part of the extracted quote. Include all forms of academic citation if they are contiguous with your selected quote.

Use ... to indicate that there is a gap of excluded text between text you chose.

For example: Text to answer... More text here... start a sentence in the middle.

No need to use the title.

Sometimes you will see authors and/or section titles. Do not use them in your answer.

Output the quote ONLY. Do not introduce it with any text, formatting, or white spaces.

If the paper is not relevant to the query and report at all, just output None
"""

USER_PROMPT_PAPER_LIST_FORMAT_EDIT = """
Here is the paper with snippets and metadata:
<paper_with_snippets>
{paper_content}
</paper_with_snippets>"""

# ============================================================================
# EDIT STEP 2: CLUSTERING/PLANNING (mirrors SYSTEM_PROMPT_QUOTE_CLUSTER)
# ============================================================================

EDIT_CLUSTER_PROMPT_DIRECTIVE = """
For each section in the current report, you need to decide an appropriate action based on the provided context:

KEEP: The section is fine as-is and doesn't need changes
REWRITE: Section needs to be modified - either to incorporate new quotes, remove papers, apply stylistic changes, or regenerate content per the edit instruction. Use this for ANY modification to an existing section.
DELETE: This section should be removed entirely
NEW: Create a new section (not in current report) with some or all of the provided quotes

For sections marked KEEP, the quotes list should be empty.
For sections marked REWRITE that incorporate new content, include the relevant quote indices.
You can also create NEW sections that weren't in the original report, but only if indicated in the instruction or intent.
If a section has to be replaced, use DELETE for the old one and NEW for the new one, rather than REWRITE, while maintaining order.
Think hard if a section with REWRITE or NEW action can lead to changes in subsequent sections (e.g., due to paper removals), you can mark those subsequent sections for REWRITE as well to ensure coherence.

**PAPER REMOVAL**:
When papers need to be removed (based on intent_analysis or edit instruction):
- Output a "papers_to_remove" list at the report level containing corpus_ids to remove
- For sections that contain removed papers, use action "REWRITE" to regenerate without those papers
- If removing papers leaves a section empty or without meaningful content, use action "DELETE"

Example when removing papers:
{{
  "papers_to_remove": ["12345", "67890"],
  "dimensions": [
    {{"name": "Methods", "format": "synthesis", "quotes": [], "action": "REWRITE"}},  // Had removed papers, needs rewrite
    {{"name": "Outdated Approaches", "format": "list", "quotes": [], "action": "DELETE"}}  // Only had removed papers
  ]
}}
"""

SYSTEM_PROMPT_QUOTE_CLUSTER_EDIT = f"""
In this task, you need to plan how to EDIT an existing report based on a user's edit instruction and the intent analysis of that instruction.

You will be provided the current report, an edit instruction with associated intent analysis, and any new quotes extracted from papers.

Your task is to create a plan for how to modify the existing report.

{EDIT_CLUSTER_PROMPT_DIRECTIVE}

You should plan modifications that:
1. Respect the existing structure and order unless the edit instruction requires changes
2. Incorporate new quotes where they fit best
3. Maintain coherence with sections that aren't being modified
4. Follow the edit instruction precisely
5. Handle paper removals: sections with removed papers need REWRITE action
6. Handle stylistic edits: use REWRITE action for the target sections

IMPORTANT: Every NEW quote should be assigned appropriately to an existing or a NEW section as per the provided edit instruction.
IMPORTANT: Sections that don't need changes should be marked with action "KEEP" and empty quotes list.
IMPORTANT: If papers_to_remove is not empty, identify which sections cite those papers and mark them for REWRITE or DELETE.
IMPORTANT: Output sections in the desired final order (clustering handles reordering naturally).

Output format:
{{{{
"cot": "Reasoning for how to edit each section based on the instruction, quotes, and intent analysis...",
"report_title": "Generate a new title for the report based on combination of the original report, edit instruction, and any new themes from the quotes. Keep it concise.",
"papers_to_remove": ["corpus_id_1", "corpus_id_2"],
"dimensions": [
  {{{{"name": "Existing Section Name", "format": "synthesis or list", "quotes": [quote indices], "action": "KEEP or REWRITE or DELETE"}}}},
  {{{{"name": "New Section Name", "format": "synthesis or list", "quotes": [quote indices], "action": "NEW"}}}},
  ...
]
}}}}
"""

USER_PROMPT_QUOTE_LIST_FORMAT_EDIT = """
Here is the edit instruction:
<edit_instruction>
{edit_instruction}
</edit_instruction>

<intent_analysis>
Papers to Add: {papers_to_add}
Papers to Remove: {papers_to_remove}
Is Stylistic Edit: {is_stylistic}
Target Sections: {target_sections}
Affects All Sections: {affects_all_sections}
</intent_analysis>

Here is the current report:
<current_report>
{current_report}
</current_report>

And here are the NEW quotes from papers that can help with the edit (may be empty if no new papers):
<quotes>
{quotes}
</quotes>"""

# ============================================================================
# EDIT STEP 3: SECTION GENERATION (mirrors PROMPT_ASSEMBLE_SUMMARY)
# ============================================================================

PROMPT_ASSEMBLE_SUMMARY_EDIT = """
A user wants to EDIT an existing report based on an edit instruction.

The edit instruction was: {edit_instruction}

Papers removed from the report (do NOT cite these): {papers_to_remove}

Here is the overall plan for the edited report:

<plan>
{plan}
</plan>

I will provide you with the name of one section from the plan at a time, along with:
1. The CURRENT content of that section (if it exists)
2. The ACTION to take (REWRITE, or NEW)
3. The list of existing quoted references already in that section (if applicable)
4. The list of new quoted references to incorporate (if applicable)

Your job is to help me write or edit this section according to the action.

Here is what has already been written in the edited report:
<already_written_sections>
{already_written}
</already_written_sections>

The section I would like you to handle next is:
<section_name>
{section_name}
</section_name>

<current_section_content>
{current_section_content}
</current_section_content>

<action>
{action}
</action>

Here are the NEW reference quotes to incorporate for this section (from new search results):
<section_references>
{section_references}
</section_references>

Here are the EXISTING citations already present in the current version of this section:
<existing_section_references>
{existing_section_references}
</existing_section_references>

<action-specific instructions>
**If action is REWRITE**: Rewrite the section content according to the edit instruction. You have two sets of references:
- section_references: NEW papers from search to incorporate
- existing_section_references: papers ALREADY cited in this section
If papers_to_remove lists corpus IDs, match them to existing_section_references by their ID and do NOT cite those papers. Retain all other existing citations where relevant, and integrate new references naturally. This covers all modifications: adding new papers, removing papers, stylistic rewrites, and restructuring.

**If action is NEW**: Write a brand new section based on the new references provided.
</action-specific instructions>

<citation instructions>
- Each reference (both new and existing) is a key value pair, where the key is a pipe separated string enclosed in square brackets representing [ID | AUTHOR_REF | YEAR | Citations: CITES].
- You can use the key to filter or retain papers based on metadata as specified in the edit instruction.

The value consists of the quote and sometimes a dictionary of inline citations referenced in that quote
eg. "[2345677 | Doe, Moe et al. | 2024 | Citations: 25]": {{"quote": "This is the reference text.", "inline citations": {{"[4517277 | Hero et al. | 2019 | Citations: 250]": "This is an inline citation."}}}}

- Please write or edit this section, making sure to cite the relevant references inline using the corresponding reference key in the format: [ID | AUTHOR_REF | YEAR | Citations: CITES]. You may use more than one reference key in a row if it's appropriate.
- Think carefully when parsing papers_to_remove. The edit instruction might specify removing only from particular sections or by certain constraints. Make sure to only exclude papers that match those criteria.
- For REWRITE actions: Use new references from section_references plus existing references from existing_section_references. If a paper appears in both, prefer the new version. Integrate naturally.

- For NEW actions: Use only the new references provided.

- Along with the quote, if any of its accompanying inline citations are relevant to or mentioned in the claim you are writing, you should cite them too using the same aforementioned format.

For example, let's say you write

"The X was shown to be Y." [1234 | A | 2023 | Citations: 3].

And the reference A (2023) you want to cite states "As shown in B (2020), X is Y." In this case, reference 1234 is textually direct support for what you wrote, but it itself clearly states that B (2020) is the actual source of this information. As such, you need to cite both, one after the other as

"The X was shown to be Y." [1234 | A | 2023 | Citations: 3] [4321 | B | 2020 | Citations: 25]

- You can add something from your own knowledge. This should only be done if you are sure about its truth and/or if there is not enough information in the references to answer the user's question. Cite the text from your knowledge as [LLM MEMORY | 2024]. The citation should follow AFTER the text. Don't cite LLM Memory with another evidence source.

- Note that all citations that support what you write must come after the text you write. That's how humans read in-line cited text. First text, then the citation.
</citation instructions>

<writing instructions>
The section should have the following characteristics:
- Before the section write a 2 sentence "TLDR;" of the section. No citations here. Precede with the text "TLDR;"
- Use direct and simple language everywhere, like "use" and "can". Avoid using more complex words if simple ones will do.
- Use the citation count to decide what is "notable" or "important". If the citation count is 100 or more, you are allowed to use value judgments like "notable."
- Some references are older. Something that claims to be "state of the art" but is from 2020 may not be any more. Please avoid making such claims.
- Be concise.
- The section you write must be coherent with already_written sections and address the edit instruction.
- For REWRITE: Maintain consistency with the existing report style. Make minimal changes to the content and structure unless the edit instruction requires otherwise. Integrate new references in a way that flows with the existing text.
- For NEW: Match the style of the overall report and the content that has already been generated.
- Multiple references may express the same idea. If so, you can cite multiple references in a single sentence.
- Do not make the same points that were made in the previous already written sections.
</writing instructions>

<format and structure instructions>
Start the section with its section_name and then a newline and then the text "TLDR;", the actual TLDR, and then write the content.
Rules for section formatting:
- For example, if the section name in the plan is "Important Papers (list)", then write it as "Important Papers" and format the section as a LIST (required).
- For example, if the section name in the plan is "Deep Dive on Networks (synthesis)" then render it as "Deep Dive on Networks" and write a SYNTHESIS paragraph (required).
- The section format MUST match what's in the parentheses of the section name. A list HAS to be a list. a SYNTHESIS has to be a paragraph. Seriously.
- Write the section content using markdown format
</format and structure instructions>
"""

# Helper template for when there are no new quotes (mirrors PROMPT_ASSEMBLE_NO_QUOTES_SUMMARY)
PROMPT_ASSEMBLE_NO_QUOTES_SUMMARY_EDIT = """
A user wants to EDIT an existing report based on an edit instruction.

The edit instruction was: {edit_instruction}

Papers removed from the report (do NOT cite these): {papers_to_remove}

Here is the overall plan for the edited report:

<plan>
{plan}
</plan>

The section I would like you to handle next is:
<section_name>
{section_name}
</section_name>

<current_section_content>
{current_section_content}
</current_section_content>

<action>
{action}
</action>

This section has no new references to incorporate from search.

Here are the EXISTING citations already present in the current version of this section:
<existing_section_references>
{existing_section_references}
</existing_section_references>

<action-specific instructions>
**If action is KEEP**: Return the current section content exactly as-is.

**If action is REWRITE**: Rewrite the section content according to the edit instruction. Use the existing_section_references to cite papers. If papers_to_remove lists corpus IDs, match them to existing_section_references by their ID and do NOT cite those papers. Retain all other existing citations where relevant.
</action-specific instructions>

<citation instructions>
- Each existing reference is a key value pair, where the key is [ID | AUTHOR_REF | YEAR | Citations: CITES] and the value contains the quote and inline citations.
- For REWRITE actions: cite existing references using their [ID | AUTHOR_REF | YEAR | Citations: CITES] keys. Do not cite papers listed in papers_to_remove.
- You can add text from your own knowledge and cite it as [LLM MEMORY | 2024]. The citation should follow AFTER the text.
- For KEEP actions with existing content: preserve any existing citations in the original content.
</citation instructions>

<writing instructions>
The section should have the following characteristics:
- Before the section write a 2 sentence "TLDR;" of the section. No citations here. Precede with the text "TLDR;"
- Use direct and simple language everywhere, like "use" and "can". Avoid using more complex words if simple ones will do.
- Be concise.
- The section must be coherent with the rest of the report and address the edit instruction.
- Maintain the voice and style of the existing report.
</writing instructions>

<format and structure instructions>
Start the section with its section_name and then a newline and then the text "TLDR;", the actual TLDR, and then expand upon it.
Rules for section formatting:
- For example, if the section name in the plan is "Important Papers (list)", then write it as "Important Papers" and format the section as a LIST (required).
- For example, if the section name in the plan is "Deep Dive on Networks (synthesis)" then render it as "Deep Dive on Networks" and write a SYNTHESIS paragraph (required).
- The section format MUST match what's in the parentheses of the section name. A list HAS to be a list. a SYNTHESIS has to be a paragraph. Seriously.
</format and structure instructions>
"""