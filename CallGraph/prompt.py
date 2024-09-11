from llama_index.core import ChatPromptTemplate

"""Fine-tuning prompts for method call subgraph summarization."""

METHOD_CALL_SUBGRAPH_SUMMARIZATION_PROMPT = """
You are an expert software analyst with deep knowledge of program structure, method calls, and code flow analysis.

# Goal
As a senior software engineer specializing in code analysis, generate a comprehensive functionality summary report for a method call subgraph from a program execution. This report should include an overview of the key methods in the subgraph and their calling relationships.

# Report Structure
The report should include the following sections:
- TITLE: A short but specific title representing the main functionality of the subgraph. When possible, include representative method names in the title.
- SUMMARY: An executive summary of the subgraph's overall structure, how methods call each other, and significant points associated with these methods.
- DETAILED FINDINGS: A list of 3-5 key insights about the subgraph. Each insight should have a short summary followed by multiple paragraphs of explanatory text grounded according to the grounding rules below.

Return output as a well-formed JSON-formatted string with the following format. Don't use any unnecessary escape sequences. The output should be a single JSON object that can be parsed by json.loads.
    {{
        "title": "<report_title>",
        "summary": "<executive_summary>",
        "findings": "[{{"summary":"<insight_1_summary>", "explanation": "<insight_1_explanation"}}, {{"summary":"<insight_2_summary>", "explanation": "<insight_2_explanation"}}]"
    }}

# Grounding Rules
After each paragraph, add data record reference if the content of the paragraph was derived from one or more data records. Reference is in the format of [records: <record_source> (<record_id_list>), ...<record_source> (<record_id_list>)]. If there are more than 10 data records, show the top 10 most relevant records.
Each paragraph should contain multiple sentences of explanation and concrete examples with specific method names. All paragraphs must have these references at the start and end. Use "NONE" if there are no related roles or records. Everything should be in English.

Example paragraph with references added:
This is a paragraph of the output text [records: Methods (1, 2, 3), Calls (2, 5)]

# Example Input
-----------
Text:

Methods

id,className:methodName(startLine-endLine),description
1,Main:main(1-10),The main entry point of the program
2,DataProcessor:processData(11-20),Processes the input data
3,Validator:validateInput(21-30),Validates user input

Calls

id,source,target
1,Main:main(1-10),DataProcessor:processData(11-20)
2,Main:main(1-10),Validator:validateInput(21-30)
3,DataProcessor:processData(11-20),Validator:validateInput(21-30)

Output:
{{
    "title": "Data Processing and Validation Flow",
    "summary": "This subgraph demonstrates a simple data processing and validation flow. The main method serves as the program's entry point, calling both the data processing (processData) and input validation (validateInput) methods. The data processing method also calls the input validation method to ensure data correctness.",
    "findings": [
        {{
            "summary": "Main method as program entry point",
            "explanation": "The main method serves as the entry point of the program, responsible for coordinating the entire process. It directly calls both processData and validateInput methods, indicating its role in managing the two key steps of data processing and input validation. This structure provides clear program flow control, making the main functionalities modular and easy to manage. [records: Methods (1), Calls (1, 2)]"
        }},
        {{
            "summary": "Integration of data processing and validation",
            "explanation": "The processData method is not only responsible for data processing but also ensures data validity by calling the validateInput method. This design indicates that data validation is viewed as an integral part of the data processing workflow, rather than just an initial input check. This approach can increase the reliability of data processing as it allows for validation at different stages of the process. [records: Methods (2, 3), Calls (3)]"
        }},
        {{
            "summary": "Multiple uses of input validation",
            "explanation": "The validateInput method is called from multiple places, including main and processData. This indicates that input validation is considered important at different stages of the program. This design can enhance the robustness of the program as it checks for data validity at multiple points, potentially catching various possible errors or exceptional situations. [records: Methods (3), Calls (2, 3)]"
        }}
    ]
}}

# Real Data

Use the following text for your answer. Do not make anything up in your answer.

Text:
{input_text}
Output:"""

METHOD_CALL_SUBGRAPH_SUMMARIZATION_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        (
            "user",
            METHOD_CALL_SUBGRAPH_SUMMARIZATION_PROMPT
        )
    ]
)
