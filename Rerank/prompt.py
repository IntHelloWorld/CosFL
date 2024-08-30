
from llama_index.core import ChatPromptTemplate

rerank_template = ChatPromptTemplate.from_messages(
    [
        (
            "user",
            "One or more test cases have failed due to a bug in the codebase.\n"
            "The possible root causes of the bug are:\n```text\n"
            "{causes_ph}\n```\n\n"
            "One of the retrieved suspicious methods in the codebase:\n```java\n"
            "{method_code_ph}\n```\n\n"
            "please carefully examine the queries and the source code to evaluate that how likely is this method "
                "the exact buggy method that responsible for the test failure.\n"
            "Your response must be in the following json format:\n"
            "{{\n"
            "\"Reason\": str, // explanation about whether this method may be responsible for the test failure\n"
            "\"Score\": float // a score in the range of [0.0, 1.0] which indicates the suspiciousness of this method\n"
            "}}"
        )
    ]
)