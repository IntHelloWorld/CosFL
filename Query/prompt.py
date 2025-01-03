
from llama_index.core import ChatPromptTemplate

single_test_query_template = ChatPromptTemplate.from_messages(
    [
        (
            "user",
            "A test case failed due to a bug in the software, the information of the failure is shown below:\n\n"
            "Code of the Failed Test:\n```java\n"
            "{test_code_ph}\n```\n\n"
            "The Output of the Failed Test:\n```text\n"
            "{test_output_ph}\n```\n\n"
            "The Error Stack Trace:\n```text\n"
            "{stack_trace_ph}\n```\n\n"
            "Given the above information, please think step by step to:\n"
            "1. Analyze the conflict between expected and actual behavior of the failed test.\n"
            "2. Understand what program elements or functionality implementation in the codebase may cause the test to fail.\n"
            "3. Generate one or more natural language queries for finding the buggy method in the codebase."
                "In each query, you can describe the functionality in the codebase that may responsible for the failed test or claim the method that you want to check."
                "Make sure that each query should ONLY concentrate on ONE functionality or method."
                "Avoid similarity between queries to ensure the diversity of the search results.\n\n"
            "Your response must be in the following json format without any extra explanation:\n"
            "{{\n"
            "  \"Conflict\": \"<the conflict between expected and actual behavior>\",\n"
            "  \"Causes\": \"<the functionality or program elements in the codebase that may cause the test to fail>\",\n"
            "  \"Queries\": [\"Find the implementation of ...\", \"Check the ... method in ... class\", ...]\n"
            "}}"
        )
    ]
)

query_merge_template = ChatPromptTemplate.from_messages(
    [
        (
            "user",
            "One or more test cases failed due to a single bug in the software,"
                " now we want to find the faulty methods in the code base with some queries,"
                " the given queries for each failed test cases are shown below:\n\n"
            "{queries_ph}"
            "Your task is to optimize the above queries based on the following rules:\n"
            "1. Merge queries that target the same functionality or method in the codebase to avoid redundancy;\n"
            "2. Keep the queries that may potentially cause all the test cases to fail to ensure the diversity;\n"
            "3. Remove the queries that are irrelevant to the failed test cases to ensure the relevance;\n"
            "4. Make sure that each query should ONLY concentrate on ONE functionality or method to avoid entanglement.\n"
            "You must directly respond with the optimized queries in the following json format without any extra explanation:\n"
            "{{\n"
            "  \"Queries\": [\"<query 1>\", \"<query 2>\"]\n"
            "}}"
        )
    ]
)


one_query_template = ChatPromptTemplate.from_messages(
    [
        (
            "user",
            "One or more test case failed due to a bug in the software, the information of the failed tests is shown below:\n\n"
            "{info_ph}"
            "Given the above information, please think step by step to:\n"
            "1. Analyze the conflict between expected and actual behavior of the failed test.\n"
            "2. Understand what program elements or functionality implementation in the codebase may cause the test to fail.\n"
            "3. Generate ONLY ONE natural language queries for finding the buggy method in the codebase."
                "In the query, you can describe the functionality in the codebase that may responsible for the failed test or claim the method that you want to check."
            "Your response must be in the following json format without any extra explanation:\n"
            "{{\n"
            "  \"Conflict\": \"<the conflict between expected and actual behavior>\",\n"
            "  \"Causes\": \"<the functionality or program elements in the codebase that may cause the test to fail>\",\n"
            "  \"Query\": \"Find the implementation of ...\"\n"
            "}}"
        )
    ]
)