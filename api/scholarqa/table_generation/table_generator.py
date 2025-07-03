from typing import List, Dict, Optional
import uuid

import itertools
from concurrent.futures import ThreadPoolExecutor

from scholarqa.table_generation.table_model import TableWidget, TableColumn, TableRow, TableCell
from scholarqa.table_generation.column_suggestion import generate_attribute_suggestions
from scholarqa.table_generation.value_generation import generate_value_suggestions
from scholarqa.utils import get_paper_metadata
from scholarqa.rag.retrieval import PaperFinder
from scholarqa.llms.constants import GPT_4o
from scholarqa.llms.litellm_helper import CostAwareLLMCaller, CostReportingArgs

class TableGenerator:
    def __init__(
        self,
        paper_finder: PaperFinder,
        llm_caller: CostAwareLLMCaller,
        max_threads: int = 1,
    ) -> None:
        self.paper_finder = paper_finder
        self.llm_caller = llm_caller
        self.max_threads = max_threads
        self.empty_cell = TableCell(id="empty", value="N/A", display_value="N/A", metadata={})
    
    def run_table_generation(
        self, 
        thread_id: str,
        user_id: str,
        original_query: str, 
        section_title: str, 
        corpus_ids: List[int],
        column_num: int = 10,
        run_subselection: bool = True,
        column_model: Optional[str] = GPT_4o,
        value_model: Optional[str] = GPT_4o,
    ) -> TableWidget:
        """
        Entry point to generate a complete table, given the original
        query sent by the user to ScholarQA, the title of the section
        under which the table will be displayed and corpus IDs of all
        papers to be included (i.e., ones cited in the section).
        """

        # Step 1: Construct a query for the column suggestion tool using
        # the section title and original user query as input. Also create
        # a cost argument object to allow the the tool to track costs
        column_suggestion_query = f"{section_title}"
        if original_query != "":
            column_suggestion_query += f", a section included in an answer to the question: {original_query}"
        cost_args = CostReportingArgs(
            task_id=thread_id,
            user_id=user_id,
            msg_id=thread_id,
            description="Cost for generating column suggestions",
            model=column_model,
        )
        output = generate_attribute_suggestions(
            corpus_ids=[str(x) for x in corpus_ids],
            query=column_suggestion_query,
            model=column_model,
            llm_caller=self.llm_caller,
            column_num=column_num,
        )
        column_cost = output.get("cost", {})
        
        # Step 2: Create a new table data structure with suggested columns.
        # While creating a column, also create requests to call the value
        # generation functionality for each column.  
        table = TableWidget(id=thread_id)
        column_suggestions = output.get("columns", [])
        value_gen_requests = []
        for column in column_suggestions:
            # Generate a UUID for the column
            column_id = str(uuid.uuid4())
            # Sometimes column names have underscores - replace them for readability
            column_name = column['name'].replace('_', ' ').title()
            table.add_columns(
                [TableColumn(
                    id=column_id,
                    name=column_name,
                    description=column["definition"],
                    is_metadata=column["is_metadata"],
                    tools=["table_cell_value_generation"],
                )]
            )
            value_gen_requests.append({
                "column_id": column_id,
                "column_name": column_name,
                "column_def": column["definition"],
                "corpus_ids": [str(x) for x in corpus_ids],
                "is_metadata": column["is_metadata"],
                "model": value_model,
                "paper_finder": self.paper_finder,
                "llm_caller": self.llm_caller,
                "cost_args": CostReportingArgs(
                    task_id=thread_id,
                    user_id=user_id,
                    msg_id=thread_id,
                    description=f"Cost for generating cell values for column {column_name}",
                    model=value_model,
                ),
            })
        
        # Step 3: Since we have corpus IDs, get paper titles and them as rows in the first
        # column of the table. Store the correspondence between corpus IDs and row IDs 
        # to use for cell creation later on.
        paper_info = self.retrieve_paper_info(corpus_ids)
        row_id_map = {}
        for corpus_id in corpus_ids:
            row_id = str(uuid.uuid4())
            row_id_map[corpus_id] = row_id
            table.add_rows([
                TableRow(
                    id=row_id,
                    display_value=paper_info[corpus_id]["title"],
                    paper_corpus_id=corpus_id,
                )
            ])

        # Step 4: Run value generation requests for all columns in parallel and add cells to the table
        all_cell_costs = []
        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            output = list(executor.map(
                self.generate_values,  
                itertools.repeat(row_id_map),
                value_gen_requests,
            ))
            for item in output:
                new_cells = item.get("cells", {})
                cell_costs = item.get("cost", {})
                table.cells.update(new_cells)
                all_cell_costs.append(cell_costs)

        if run_subselection:
            table = self.subselect_columns_and_rows(table)
        final_cost_dict = {
            "column_cost": column_cost,
            "cell_cost": all_cell_costs,
        }

        return table, final_cost_dict
    
    """
    Functions to only select a subset of informative 
    and non-redundant columns and rows from the table
    """
    def column_to_doc(self, column_id: str, table: TableWidget):
        cell_ids = [f"{row.id}_{column_id}" for row in table.rows]
        cells = [table.cells.get(cell_id, self.empty_cell).value for cell_id in cell_ids]
        cells = [cell for cell in cells if (cell != None and len(cell) > 0 and cell != "N/A")]
        return {'valid_cells': len(cells), 'doc': ' ||| '.join(cells)}
        # return {'valid_cells': len(cells), 'doc': ' ||| '.join(map(clean, cells))}

    def row_to_doc(self, row_id: str, table: TableWidget):
        cell_ids = [f"{row_id}_{column.id}" for column in table.columns]
        cells = [table.cells.get(cell_id, self.empty_cell).value for cell_id in cell_ids]
        cells = [cell for cell in cells if (cell != None and len(cell) > 0 and cell != "N/A")]
        return {'valid_cells': len(cells), 'doc': ' ||| '.join(cells)}
        # return {'valid_cells': len(cells), 'doc': ' ||| '.join(map(clean, cells))}

    def keep_rows(self, table: TableWidget, row_ids: List[str]):
        table.rows = [row for row in table.rows if row.id in row_ids]
        table.cells = {cell_id: cell for cell_id, cell in table.cells.items() if cell_id.split('_')[0] in row_ids}
        return table

    def keep_columns(self, table: TableWidget, column_ids: List[str]):
        table.columns = [column for column in table.columns if column.id in column_ids]
        table.cells = {cell_id: cell for cell_id, cell in table.cells.items() if cell_id.split('_')[1] in column_ids}
        return table


    def subselect_columns_and_rows(self, original_table: TableWidget, max_rows=6, max_columns=6):
        table = original_table.model_copy(deep=True)

        row_valid_cells = [{'row_id': row.id, 'valid_cells': self.row_to_doc(row.id, table)['valid_cells']} for row in table.rows]
        row_valid_cells = [row for row in row_valid_cells if row['valid_cells'] >= max_columns]
        row_valid_cells = sorted(row_valid_cells, key=lambda x: x['valid_cells'], reverse=True)
        row_valid_cells = row_valid_cells[:max_columns*2]
        
        table = self.keep_rows(table, [row['row_id'] for row in row_valid_cells])

        column_valid_cells = [{'column_id': column.id, **self.column_to_doc(column.id, table)} for column in table.columns]
        column_valid_cells = [column for column in column_valid_cells if column['valid_cells'] > len(table.rows)*0.7]

        column_valid_cells = column_valid_cells[:max_columns]

        table =self.keep_columns(table, [column['column_id'] for column in column_valid_cells])

        row_valid_cells = [{'row_id': row.id, 'valid_cells': self.row_to_doc(row.id, table)['valid_cells']} for row in table.rows]
        row_valid_cells = [row for row in row_valid_cells if row['valid_cells'] >= max_rows / 2]
        row_valid_cells = sorted(row_valid_cells, key=lambda x: x['valid_cells'], reverse=True)
        row_valid_cells = row_valid_cells[:max_rows]
        
        table = self.keep_rows(table, [row['row_id'] for row in row_valid_cells])
        return table

    def retrieve_paper_info(self, corpus_ids: List[str]) -> Dict:
        """ 
        Given a set of corpus IDs for papers to be added to the table,
        retrieve paper titles using the Semantic Scholar batch querying API.
        """
        paper_metadata = get_paper_metadata(corpus_ids)
        paper_info = {corpus_id: paper_metadata[str(corpus_id)] if str(corpus_id) in paper_metadata else {} for corpus_id in corpus_ids}
        return paper_info

    def generate_values(self, row_id_map: dict, request: dict):
        """
        Given a request to generate cell values for a column, call the value 
        generation functionality and create and return TableCell objects for each.
        """
        column_id = request.pop("column_id")
        output = generate_value_suggestions(**request)
        generated_values = output.get("cell_values", [])
        cell_costs = output.get("cost", {})
        table_cells = {}
        for value in generated_values:
            cell_id = f"{row_id_map[int(value['corpusId'])]}_{column_id}"
            cell = TableCell(
                id=cell_id,
                value=value['displayValue'],
                display_value=value['displayValue'],
                metadata=value.get('metadata', None),
            )
            table_cells[cell_id] = cell
        output ={
            "cells": table_cells,
            "cost": cell_costs, 
        }
        return output