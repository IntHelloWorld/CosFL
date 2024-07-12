
from llama_index.core import ChatPromptTemplate

single_test_analysis_template = ChatPromptTemplate.from_messages(
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
            "3. Generate one or more queries for finding the buggy method in the codebase."
                "In each query, you can describe the functionality in the codebase that may responsible for the failed test or claim the method that you want to check."
                "Make sure that each query should ONLY concentrate on ONE functionality or method."
                "Avoid similarity between queries to ensure the diversity of the search results.\n\n"
            "Your response must be in the following json format:\n"
            "{{\n"
            "  \"Conflict\": \"<the conflict between expected and actual behavior>\",\n"
            "  \"Causes\": \"<the functionality or program elements in the codebase that may cause the test to fail>\",\n"
            "  \"Queries\": [\"<query 1>\", \"<query 2>\", ...]\n"
            "}}"
        )
    ]
)

query_merge_template = ChatPromptTemplate.from_messages(
    [
        (
            "user",
            "Multiple test cases failed due to a single bug in the software, the queries for each failed test cases are shown below:\n\n"
            "{queries_ph}"
            "Your task is to process the above queries based on the following rules:\n"
            "1. Merge queries that target the same functionality or method in the codebase to avoid redundancy.\n"
            "2. Keep the queries that may potentially cause all the test cases to fail to ensure the diversity.\n"
            "3. Remove the queries that are irrelevant to the failed test cases to ensure the relevance.\n"
            "4. Make sure that each query should ONLY concentrate on ONE functionality or method.\n\n"
            "Your response must be in the following json format:\n"
            "{{\n"
            "  \"Queries\": [\"<query 1>\", \"<query 2>\", ...],\n"
            "}}"
        )
    ]
)


# cause_merge_template = ChatPromptTemplate.from_messages(
#     [
#         (
#             "user",
#             "Multiple test cases failed due to a single bug in the software, the possible causes for each failed test cases are shown below:\n\n"
#             "{causes_ph}"
#             "Your task is to merge the above causes to ONE paragraph which concludes the possible root causes of the bug.\n"
#             "Pleare directly provide the merged causes without any extra explanation."
#         )
#     ]
# )