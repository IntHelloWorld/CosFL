
from llama_index.core import ChatPromptTemplate

CHAT_RERANK_PROMPT = """
A bug in the codebase has caused one or more test cases to fail. Your task is to analyze a potentially suspicious method and determine its likelihood of being the source of the bug.

Given information:
Faulty Location Hypothesis:

{causes_ph}

One of the retrieved suspicious methods in the codebase:

{method_code_ph}

Instructions:
1. Carefully examine the provided Faulty Location Hypothesis and the source code of the suspicious method.
2. Consider the following factors in your analysis:
   - How closely does the method's functionality align with the Faulty Location Hypothesis?
   - Are there any obvious issues or potential bugs in the method's implementation?
   - Does the method interact with components or data that could be related to the test failures?
   - Are there any error-prone patterns or anti-practices present in the code?

3. Evaluate the likelihood that this method is responsible for the test failure.
4. Provide a concise but thorough explanation for your assessment.
5. Assign a suspiciousness score to the method on a scale of 0.0 to 10.0, where:
   - 0.0 means the method is very unlikely to be the cause of the bug
   - 10.0 means the method is very likely to be the cause of the bug

The output should include the following sections:
- REASON: A clear, concise explanation of your assessment, including key points from your analysis.
- SCORE: A suspiciousness score between 0.0 and 10.0, based on your evaluation.

Return output as a well-formed JSON-formatted string with the following format. Don't use any unnecessary escape sequences. The output should be a single JSON object that can be parsed by json.loads.
  {{
    "Reason": "<detailed_explanation>",
    "Score": <suspicousness_score>
  }}
  
Example JSON output:
  {{
    "Reason": "The method closely matches the faulty location hypothesis and contains a potential bug in the error handling logic.",
    "Score": 8.5
  }}

Note: Ensure your explanation in the "Reason" field is detailed enough to justify the score you've assigned.
"""

EXAMPLE_CHAT_RERANK_PROMPT = """
A bug in the codebase has caused one or more test cases to fail. Your task is to analyze a potentially suspicious method and determine its likelihood of being the source of the bug.

Given information:
Faulty Location Hypothesis:

{
  \"context\": \"The bug likely resides in the component responsible for user authentication and session management.\",
  \"functionality\": \"The method responsible for session token validation process is not correctly handling expired tokens, leading to unauthorized access to protected resources.\",
  \"logic\": \"The buggy code is likely not properly checking the expiration time of the token. It may be using an incorrect comparison operator or not accounting for time zone differences when comparing the current time with the token's expiration timestamp.\"
}

One of the retrieved suspicious methods in the codebase:

 396 public void test2947660() {
 397         AbstractCategoryItemRenderer r = new LineAndShapeRenderer();
 398         assertNotNull(r.getLegendItems());
 399         assertEquals(0, r.getLegendItems().getItemCount());
 400 
 401         DefaultCategoryDataset dataset = new DefaultCategoryDataset();
 402         CategoryPlot plot = new CategoryPlot();
 403         plot.setDataset(dataset);
 404         plot.setRenderer(r);
 405         assertEquals(0, r.getLegendItems().getItemCount());
 406 
 407         dataset.addValue(1.0, \"S1\", \"C1\");
 408         LegendItemCollection lic = r.getLegendItems();
 409         assertEquals(1, lic.getItemCount());
 410         assertEquals(\"S1\", lic.get(0).getLabel());

Instructions:
1. Carefully examine the provided Faulty Location Hypothesis and the source code of the suspicious method.
2. Consider the following factors in your analysis:
   - How closely does the method's functionality align with the Faulty Location Hypothesis?
   - Are there any obvious issues or potential bugs in the method's implementation?
   - Does the method interact with components or data that could be related to the test failures?
   - Are there any error-prone patterns or anti-practices present in the code?

3. Evaluate the likelihood that this method is responsible for the test failure.
4. Provide a concise but thorough explanation for your assessment.
5. Assign a suspiciousness score to the method on a scale of 0.0 to 10.0, where:
   - 0.0 means the method is very unlikely to be the cause of the bug
   - 10.0 means the method is very likely to be the cause of the bug

The output should include the following sections:
- REASON: A clear, concise explanation of your assessment, including key points from your analysis.
- SCORE: A suspiciousness score between 0.0 and 10.0, based on your evaluation.

Return output as a well-formed JSON-formatted string with the following format. Don't use any unnecessary escape sequences. The output should be a single JSON object that can be parsed by json.loads.
  {{
    "Reason": "<detailed_explanation>",
    "Score": <suspicousness_score>
  }}
  
Example JSON output:
  {{
    "Reason": "The method closely matches the faulty location hypothesis and contains a potential bug in the error handling logic.",
    "Score": 8.5
  }}

Note: Ensure your explanation in the "Reason" field is detailed enough to justify the score you've assigned.
"""


EXAMPLE_RERANK_RESPONSE = """
{{
    "Reason": "The method closely matches the faulty location hypothesis and contains a potential bug in the error handling logic.",
    "Score": 8.5
}}
"""

CHAT_RERANK_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        (
            "user",
            CHAT_RERANK_PROMPT
        )
    ]
)