"""
Edit-specific prompts that extend the existing SQA prompts.
These prompts mirror the structure of the originals but include edit context.
"""

# ============================================================================
# STEP 1: QUOTE EXTRACTION (mirrors SYSTEM_PROMPT_QUOTE_PER_PAPER)
# ============================================================================

SYSTEM_PROMPT_QUOTE_PER_PAPER_EDIT = """
In this task, you are presented with an EDIT INSTRUCTION for an existing report, and an academic paper with snippets and metadata.

The user wants to modify their existing report. Your job is to extract relevant quotes from this paper that will help fulfill the edit instruction.

<current_report_context>
The current report has the following sections:
{current_sections_summary}
</current_report_context>

<edit_instruction>
{edit_instruction}
</edit_instruction>

Stitch together text from the paper content that is relevant to the edit instruction.

To be clear, copy EXACT text ONLY.

Include any references that are part of the text to be copied. The references can occur at the beginning, middle, or end of the text.

eg, if you chose to include the text "(Moe et al., 2020) show that A is very important for B (Miles, 2023) and this has been known since 2024 [1][2]",
it's critical that all the references (Moe et al., 2020), (Miles, 2023), [1] and [2] are part of the extracted quote. Include all forms of academic citation if they are contiguous with your selected quote.

Use ... to indicate that there is a gap of excluded text between text you chose.

For example: Text to answer... More text here... start a sentence in the middle.

No need to use the title.

Sometimes you will see authors and/or section titles. Do not use them in your answer.

Output the quote ONLY. Do not introduce it with any text, formatting, or white spaces.

If the paper does not help with the edit instruction at all, just output None
"""

USER_PROMPT_PAPER_LIST_FORMAT_EDIT = """
Here is the edit instruction:<edit_instruction>
{edit_instruction}
</edit_instruction>

Here is the current report context:<current_report>
{current_report_summary}
</current_report>

And here is the paper with snippets and metadata that may help with the edit:
<paper_with_snippets>
{paper_content}
</paper_with_snippets>"""

# ============================================================================
# STEP 2: CLUSTERING/PLANNING (mirrors SYSTEM_PROMPT_QUOTE_CLUSTER)
# ============================================================================

EDIT_CLUSTER_PROMPT_DIRECTIVE = """
For each section in the current report, you need to decide what to do:

KEEP: The section is fine as-is and doesn't need changes
EXPAND: Add more content to the section with the new quotes
ADD_TO: Add the new papers/quotes to this existing section
REPLACE: Replace this section's content with new content based on the quotes
DELETE: This section should be removed
NEW: Create a new section (not in current report) with these quotes

For sections marked KEEP, the quotes list should be empty.
For sections with actions that modify content, include the relevant quote indices.
You can also create NEW sections that weren't in the original report.
"""

SYSTEM_PROMPT_QUOTE_CLUSTER_EDIT = f"""
In this task, you are presented with quoted passages that were collected from a set of papers, and you need to plan how to EDIT an existing report based on a user's edit instruction.

<current_report>
{{current_report}}
</current_report>

<edit_instruction>
{{edit_instruction}}
</edit_instruction>

<new_quotes>
The following are NEW quotes extracted from papers (some may be user-mentioned, some from search):
{{quotes}}
</new_quotes>

Your task is to create a plan for how to modify the existing report using these new quotes.

{EDIT_CLUSTER_PROMPT_DIRECTIVE}

You should plan modifications that:
1. Respect the existing structure unless the edit instruction requires changes
2. Incorporate new quotes where they fit best
3. Maintain coherence with sections that aren't being modified
4. Follow the edit instruction precisely

IMPORTANT: Every NEW quote should be assigned to either an existing section (to expand/add to it) or a new section you create.
IMPORTANT: Sections that don't need changes should be marked with action "KEEP" and empty quotes list.
IMPORTANT: The output structure should mirror the existing ClusterPlan format but with action annotations.

Output format:
{{
"cot": "Reasoning for how to edit each section based on the instruction and new quotes...",
"report_title": "Keep existing title or provide updated title if instruction requires it",
"dimensions": [
  {{"name": "Existing Section Name", "format": "synthesis or list", "quotes": [quote indices], "action": "KEEP or EXPAND or ADD_TO or REPLACE or DELETE"}},
  {{"name": "New Section Name", "format": "synthesis or list", "quotes": [quote indices], "action": "NEW"}},
  ...
]
}}
"""

USER_PROMPT_QUOTE_LIST_FORMAT_EDIT = """
Here is the edit instruction:
<edit_instruction>
{edit_instruction}
</edit_instruction>

Here is the current report:
<current_report>
{current_report}
</current_report>

And here are the NEW quotes from papers that can help with the edit:
<quotes>
{quotes}
</quotes>"""

# ============================================================================
# STEP 3: SECTION GENERATION (mirrors PROMPT_ASSEMBLE_SUMMARY)
# ============================================================================

PROMPT_ASSEMBLE_SUMMARY_EDIT = """
A user wants to EDIT an existing report based on an edit instruction.

The edit instruction was: {edit_instruction}

Here is the overall plan for the edited report:

<plan>
{plan}
</plan>

I will provide you with the name of one section from the plan at a time, along with:
1. The CURRENT content of that section (if it exists)
2. The ACTION to take (KEEP, EXPAND, ADD_TO, REPLACE, DELETE, or NEW)
3. The list of new quoted references to incorporate (if applicable)

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

Here are the new reference quotes to incorporate for this section:
<section_references>
{section_references}
</section_references>

<action-specific instructions>
**If action is KEEP**: Return the current section content exactly as-is.

**If action is EXPAND**: Keep all current content and add NEW paragraphs that incorporate the new references. The new content should flow naturally from the existing content.

**If action is ADD_TO**: Integrate the new references into the existing narrative. Weave them in naturally, adding new sentences or paragraphs as needed.

**If action is REPLACE**: Write entirely new content based on the new references, replacing the current content completely.

**If action is NEW**: Write a brand new section based on the references provided.

**If action is DELETE**: (You won't see this - deleted sections are skipped)
</action-specific instructions>

<citation instructions>
- Each reference is a key value pair, where the key is a pipe separated string enclosed in square brackets representing [ID | AUTHOR_REF | YEAR | Citations: CITES].

The value consists of the quote and sometimes a dictionary of inline citations referenced in that quote
eg. "[2345677 | Doe, Moe et al. | 2024 | Citations: 25]": {{"quote": "This is the reference text.", "inline citations": {{"[4517277 | Hero et al. | 2019 | Citations: 250]": "This is an inline citation."}}}}

- Please write or edit this section, making sure to cite the relevant references inline using the corresponding reference key in the format: [ID | AUTHOR_REF | YEAR | Citations: CITES]. You may use more than one reference key in a row if it's appropriate.

- For EXPAND and ADD_TO actions: Keep existing citations in place. Add new citations for the new content.

- For REPLACE and NEW actions: Use only the new references provided.

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
- For EXPAND/ADD_TO: Maintain the voice and style of the existing content.
- For REPLACE/NEW: Match the style of the overall report.
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

This section has no new references to incorporate. Handle it based on the action and your knowledge.

<action-specific instructions>
**If action is KEEP**: Return the current section content exactly as-is.

**If action is EXPAND** or **REPLACE**: Use your knowledge to expand or rewrite the section according to the edit instruction.
</action-specific instructions>

<citation instructions>
- Cite text as [LLM MEMORY | 2024]. The citation should follow AFTER the text.
- For KEEP or EXPAND actions with existing content: preserve any existing citations in the original content.
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
