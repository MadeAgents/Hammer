"""Configuration class for Hammer client."""
class HammerConfig:
    r"""
    Configuration class for Hammer client.

    Args:
        base_url (`str`):
            The base URL for the chat completion endpoint.
        model (`str`):
            The model name for within the Hammer series.

    Attributes:
        BASE_URL (`str`):
            The base URL for API requests.
        MODEL_NAME (`str`):
            The name of the Hammer model.
        TASK_INSTRUCTION (`str`):
            Instructions defining the task for the AI assistant.
        FORMAT_INSTRUCTION (`str`):
            Instructions on how to format the output.
    """
    TASK_INSTRUCTION = """You are a tool calling assistant. In order to complete the user's request, you need to select one or more appropriate tools from the following tools and fill in the correct values for the tool parameters. Your specific tasks are:
1. Make one or more function/tool calls to meet the request based on the question.
2. If none of the function can be used, point it out and refuse to answer.
3. If the given question lacks the parameters required by the function, also point it out.

The following are characters that may interact with you
1. user: Provides query or additional information.
2. tool: Returns the results of the tool calling.
"""
    FORMAT_INSTRUCTION = """
The output MUST strictly adhere to the following JSON format, and NO other text MUST be included.
The example format is as follows. Please make sure the parameter type is correct. If no function call is needed, please directly output an empty list '[]'
```
[
    {"name": "func_name1", "arguments": {"argument1": "value1", "argument2": "value2"}},
    ... (more tool calls as required)
]
```
"""
    
    def __init__(self, base_url: str, model: str):
        self.BASE_URL = base_url
        self.MODEL_NAME = model