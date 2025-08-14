# step 1 prompt
SYSTEM_PROMPT_QUOTE_PER_PAPER = """
In this task, you are presented with a user query and an academic paper with snippets and metadata.

Stitch together text from the paper content to directly answer the question. 

To be clear, copy EXACT text ONLY.

Include any references that are part of the text to be copied. The references can occur at the beginning, middle, or end of the text.

eg, if you chose to include the text "(Moe et al., 2020) show that A is very important for B (Miles, 2023) and this has been known since 2024 [1][2]", 
it's critical that all the references (Moe et al., 2020), (Miles, 2023), [1] and [2] are part of the extracted quote. Include all forms of academic citation if they are contiguous with your selected quote.

Use ... to indicate that there is a gap of excluded text between text you chose.

For example: Text to answer... More text here... start a sentence in the middle.

No need to use the title. 

Sometimes you will see authors and/or section titles. Do not use them in your answer.

Output the quote ONLY. Do not introduce it with any text, formatting, or white spaces.

If the paper does not answer the user query at all, just output None
"""

USER_PROMPT_PAPER_LIST_FORMAT = """
Here is the user's query:<user_query>
{}
</user_query>
And here is the paper with snippets and metadata that may have salient content for the query:
<paper_with_snippets>
{}
</paper_with_snippets>"""

# step 2 prompts
CLUSTER_PROMPT_FEW_SHOTS = """For example, if the user query is "Is true that: Language models are not universally better at discriminating among previously generated alternatives than generating initial responses."
Then the DIMENSIONS could be "language models studied", "discrimination approaches", "discrimination performance", etc

Another example, if the user query is "What are alternatives to the ADAM algorithm?" 
Then you can either have a single dimension: "Alternatives to ADAM" or you can have multiple dimensions like "Alternatives to ADAM in Optimization", "Alternatives to ADAM in Deep Learning". 
"""

CLUSTER_PROMPT_DIRECTIVE = """For each section, decide if it should be a bullet-point list or a synthesis paragraph. 
Bullet-point lists are right when the user wants a list or table of items. 
Synthesis paragraphs are right when the user wants a coherent explanation or comparison or analysis or singular answer.

For user queries that are simpler, you only need one dimension. 
For example, if the query is "I want a list of alternatives to GPT-3" or "Give me a table of GPT-3 alternatives", then there is only dimension "List of Alternatives to GPT-3".
"""

SYSTEM_PROMPT_QUOTE_CLUSTER = f"""
In this task, you are presented with quoted passages that were collected from a set of papers.
The goal is to fuse all those quotes into a summary answer/report that satisfies the user query in an easy-to-consume format.

As an intermediate step, please cluster quotes for different section in the summary. 
First, output a set of DIMENSIONS that break down the user query for a scientific audience. 

{CLUSTER_PROMPT_FEW_SHOTS}

You should start by making a plan of which candidate dimensions make sense to clearly and directly answer the query, ignoring the snippets.

The first section should be title "Introduction" or "Background" to provide the user the key basics needed to understand the rest of the answer.

{CLUSTER_PROMPT_DIRECTIVE}

IMPORTANT: Make sure the clusters are in the same order you would then write the corresponding summary.
IMPORTANT: Make sure that EVERY input quote is included somewhere in the output.
IMPORTANT: Some sections may not have any quotes to support them. Include them in the output JSON regardless with an empty list. This is particularly true of the "Introduction" or "Background" section.

Choose list or synthesis with deep care and wisdom. Start with a markdown justification for the section name and its format.

The last thing you output is an assignment of each quote to a dimension like so:  
{{
"cot": "Justification for every dimension name and its format...",
"dimensions": [{{"name": "dimension name 1", "format": "synthesis or list", "quotes": [comma-delimited highlights indices]}},
{{"name": "dimension name 2", "format": "synthesis or list", "quotes": [comma-delimited highlights indices]}},
{{"name": "dimension name 3", "format": "synthesis or list", "quotes": []}},  # empty because we didn't find any supporting quotes
...
]
}}
Dont add unnecessary linebreaks or spaces to the output.
"""

USER_PROMPT_QUERY_FORMAT = """
Here is the user's query:
    <user_query>
    {}
    </user_query>
"""

USER_PROMPT_QUOTE_LIST_FORMAT = USER_PROMPT_QUERY_FORMAT + """And here are the quotes from the papers that may have salient content for the query:
    <quotes>
    {}
    </quotes>"""

# step 3 prompt
PROMPT_ASSEMBLE_SUMMARY = """
A user issued a query and a set of papers were provided with salient content.
The user query was: {query}

Here is the overall plan for the summary answer:

<plan>
{plan}
</plan>

I will provide you with the name of one section from the plan at a time, along with the list of chosen quotes for that section. 
Your job is to help me write this section and cite the provided quoted references. 

Here is what has already been written:
<already_written_sections>
{already_written}
</already_written_sections>

The section I would like you to write next is (type of section):
<section_name>
{section_name}
</section_name>

Here are the reference quotes to cite for this section:
<section_references>
{section_references}
</section_references>

<citation instructions>
- Each reference is a key value pair, where the key is a pipe separated string enclosed in square brackets representing [ID | AUTHOR_REF | YEAR | Citations: CITES].

The value consists of the quote and sometimes a dictionary of inline citations referenced in that quote
eg. "[2345677 | Doe, Moe et al. | 2024 | Citations: 25]": {{"quote": "This is the reference text.", "inline citations": {{"[4517277 | Hero et al. | 2019 | Citations: 250]": "This is an inline citation."}}}}

- Please write this section, making sure to cite the relevant references inline using the corresponding reference key in the format: [ID | AUTHOR_REF | YEAR | Citations: CITES]. You may use more than one reference key in a row if it's appropriate. In general, use all of the references that support your written text.

- Along with the quote, if any of its accompanying inline citations are relevant to or mentioned in the claim you are writing, you should cite them too using the same aforementioned format. 

For example, let's say you write 

"The X was shown to be Y." [1234 | A | 2023 | Citations: 3]. 

And the reference A (2023) you want to cite states "As shown in B (2020), X is Y." In this case, reference 1234 is textually direct support for what you wrote, but it itself clearly states that B (2020) is the actual source of this information. As such, you need to cite both, one after the other as 

"The X was shown to be Y." [1234 | A | 2023 | Citations: 3] [4321 | B | 202 | Citations: 25]

- You can add something from your own knowledge. This should only be done if you are sure about its truth and/or if there is not enough information in the references to answer the user's question. Cite the text from your knowledge as [LLM MEMORY | 2024]. The citation should follow AFTER the text. Don't cite LLM Memory with another evidence source. 

- Note that all citations that support what you write must come after the test you write. That's how humans read in-line cited text. First text, then the citation.
</citation instructions>

<writing instructions>
The section should have the following characteristics:
- Before the section write a 2 sentence "TLDR;" of the section. No citations here. Precede with the text "TLDR;"
- Use direct and simple language everywhere, like "use" and "can". Avoid using more complex words if simple ones will do.
- Use the citation count to decide what is "notable" or "important". If the citation count is 100 or more, you are allowed to use value judgments like "notable."
- Some references are older. Something that claims to be "state of the art" but is from 2020 may not be any more. Please avoid making such claims.
- Be concise.
- The section you write must be a coherent continuation of already_written and directly answer the user query.
- The section you write should add to the user's understanding of what they learned in already_written. 
- Multiple references may express the same idea. If so, you can cite multiple references in a single sentence.
- Do not make the same points that were made in the previous already written sections.
</writing instructions>

<format and structure instructions>
Start the section with its section_name and then a newline and then the text "TLDR;", the actual TLDR, and then write the summary. 
Rules for section formatting:
- For example, if the section name in the plan is "Important Papers (list)", then write it as "Important Papers" and format the section as a LIST (required). 
- For example, if the section name in the plan is "Deep Dive on Networks (synthesis)" then render it as "Deep Dive on Networks" and write a SYNTHESIS paragraph (required).
- The section format MUST match what's in the parentheses of the section name. A list HAS to be a list. a SYNTHESIS has to be a paragraph. Seriously.
- Write the section content using markdown format
</format and structure instructions>
"""

PROMPT_ASSEMBLE_NO_QUOTES_SUMMARY = """
A user issued a query and the goal is to provide an answer based on the given below plan.
The user query was: {query}

Here is the overall plan for the summary answer:

<plan>
{plan}
</plan>

I will provide you with the name of one section from the plan at a time. 
Your job is to help me write this section. 

Here is what has already been written:
<already_written_sections>
{already_written}
</already_written_sections>

The section I would like you to write next is (type of section):
<section_name>
{section_name}
</section_name>


<citation instructions>
- Please write this section. This should only be done if you are sure about its truth.
- Cite the text as [LLM MEMORY | 2024]. The citation should follow AFTER the text.
</citation instructions>

<writing instructions>
The section should have the following characteristics:
- Before the section write a 2 sentence "TLDR;" of the section. No citations here. Precede with the text "TLDR;"
- Use direct and simple language everywhere, like "use" and "can". Avoid using more complex words if simple ones will do.
- Be concise.
- The section you write must be a coherent continuation of already_written and directly answer the user query.
- The section you write should add to the user's understanding of what they learned in already_written. 
- Multiple references may express the same idea. If so, you can cite multiple references in a single sentence.
- Do not make the same points that were made in the previous already written sections.
- Fix weird unicode.
</writing instructions>

<format and structure instructions>
Start the section with its section_name and then a newline and then the text "TLDR;", the actual TLDR, and then expand upon it. 
Rules for section formatting:
- For example, if the section name in the plan is "Important Papers (list)", then write it as "Important Papers" and format the section as a LIST (required). 
- For example, if the section name in the plan is "Deep Dive on Networks (synthesis)" then render it as "Deep Dive on Networks" and write a SYNTHESIS paragraph (required).
- The section format MUST match what's in the parentheses of the section name. A list HAS to be a list. a SYNTHESIS has to be a paragraph. Seriously.
</format and structure instructions>
"""

QUERY_DECOMPOSER_PROMPT = """
<task>
Your task is to analyze a query issued by a user of an academic question-answering system and break it down into parts relevant for searching and retrieving high-quality, well-cited answers. The goal is to create a structured JSON output that can be used by an academic search engine API, which supports filters like publication years, venues, authors, and fields of study.

Your output should decompose the query into the following components:
1. Publication years: If the query specifies a time range (e.g., “recent” or “last five years”), convert it to the relevant year range. If no time range is mentioned, leave these fields blank.  
   - The current year is 2025. Interpret “recent” as 2022–2025 and adjust other relative terms (e.g., “last decade,” “since 2018”) accordingly.
2. Venues: Include any journals, conferences, or publishers mentioned explicitly in the query as a comma separated string. Use their exact names.
3. Authors: List any authors mentioned explicitly in the query. Each author name should appear as a separate entry in an array.
4. Fields of study: Use only the following fields of study. If the query includes subfields or ambiguous terms, map them to the closest match from this list:  
   - Computer Science, Medicine, Chemistry, Biology, Materials Science, Physics, Geology, Psychology, Art, History, Geography, Sociology, Business, Political Science, Economics, Philosophy, Mathematics, Engineering, Environmental Science, Agricultural and Food Sciences, Education, Law, Linguistics.  
   Concatenate multiple fields with commas and no spaces (e.g., `Physics,Mathematics`).
5. Rewritten query: Simplify the remaining query into a concise, natural-language phrase, excluding any information already extracted into `year`, `venues`, `authors`, or `fields_of_study`.
6. Rewritten query for keyword search: Remove unnecessary stop words and connectors to create a keyword-friendly version of the remaining query, excluding any information already extracted into other fields.
7. Complex, multi-sentence queries should still be complete in terms of content when rewritten. The goal of the rewritten query is to remove the metadata like year, venues, authors and fields_of_study, but keep all of the important topical content that needs to be addressed in an answer.

<note about handling ambiguous terms and missing information>  
- If a field cannot be inferred from the query (e.g., no authors are mentioned), leave it empty.  
- For terms not matching the fields of study list, map them to the closest matching field(s). For instance, “machine learning” should map to `Computer Science`, while “neuroscience” might map to `Biology` or `Psychology` based on context.  
</note about handling ambiguous terms and missing information>  
</task>

<examples>

<example input #1>  
What are the latest papers by Andrew Ng on deep reinforcement learning?  
</example input #1>

<example output #1>  
```json
{
    "earliest_search_year": "2022",
    "latest_search_year": "2025",
    "venues": "",
    "authors": ["Andrew Ng"],
    "field_of_study": "Computer Science",
    "rewritten_query": "Deep reinforcement learning.",
    "rewritten_query_for_keyword_search": "deep reinforcement learning"
}
```
</example output #1>  

<example input #2>  
Summarize the findings on climate policy impacts in articles published in Nature or Science from the last five years.  
</example input #2>

<example output #2>  
```json
{
    "earliest_search_year": "2020",
    "latest_search_year": "2025",
    "venues": "Nature,Science",
    "authors": [],
    "field_of_study": "Environmental Science,Political Science",
    "rewritten_query": "Findings on climate policy impacts.",
    "rewritten_query_for_keyword_search": "climate policy impacts"
}
```
</example output #2>  

<example input #3>  
Discuss recent contributions by Noam Chomsky to the study of linguistics and cognitive science.  
</example input #3>

<example output #3>  
```json
{
    "earliest_search_year": "2022",
    "latest_search_year": "2025",
    "venues": "",
    "authors": ["Noam Chomsky"],
    "field_of_study": "Linguistics,Psychology",
    "rewritten_query": "Contributions to linguistics and cognitive science.",
    "rewritten_query_for_keyword_search": "linguistics cognitive science"
}
```
</example output #3>  

<example input #4>  
What are the effects of climate change on agricultural productivity in Sub-Saharan Africa in recent studies?
</example input #4>

<example output #4>  
```json
{
    "earliest_search_year": "2022",
    "latest_search_year": "2025",
    "venues": "",
    "authors": [],
    "field_of_study": "Environmental Science,Agricultural and Food Sciences",
    "rewritten_query": "Effects of climate change on agricultural productivity in Sub-Saharan Africa.",
    "rewritten_query_for_keyword_search": "climate change agricultural productivity Sub-Saharan Africa"
}
```
</example output #4>  

<example input #5>  
Explore the role of neural networks in solving mathematical optimization problems. 
</example input #5>

<example output #5>  
```json
{
    "earliest_search_year": "",
    "latest_search_year": "",
    "venues": "",
    "authors": [],
    "field_of_study": "Computer Science,Mathematics",
    "rewritten_query": "Role of neural networks in solving mathematical optimization problems.",
    "rewritten_query_for_keyword_search": "neural networks mathematical optimization problems"
}
```
</example output #5>  

<example input #6>  
Discuss the historical significance of the Renaissance period on modern art and philosophy.
</example input #6>

<example output #6>  
```json
{
    "earliest_search_year": "",
    "latest_search_year": "",
    "venues": "",
    "authors": [],
    "field_of_study": "Art,Philosophy",
    "rewritten_query": "Historical significance of Renaissance on modern art and philosophy.",
    "rewritten_query_for_keyword_search": "Renaissance modern art philosophy"
}
```
</example output #6>  

<example input #7>  
Review papers by Andrew Ng and Yann LeCun on neural networks since 2010. 
</example input #7>

<example output #7>  
```json
{
    "earliest_search_year": "2010",
    "latest_search_year": "2025",
    "venues": "",
    "authors": ["Andrew Ng", "Yann LeCun"],
    "field_of_study": "Computer Science",
    "rewritten_query": "Neural networks.",
    "rewritten_query_for_keyword_search": "neural networks"
}
```
</example output #7> 
</examples>
"""
