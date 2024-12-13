import json
from openai import OpenAI
from client.config import HammerConfig

class HammerChatCompletion:
    r"""
    A class for handling chat completions using the Hammer model.

    Args:
        base_url (`str`):
            The base URL for the API endpoint.
        model_name (`str`):
            The name of the Hammer model to use.
        task_instruction (`str`):
            Instructions defining the task for the model.
        format_instruction (`str`):
            Instructions on how to format the output.

    Attributes:
        model_name (`str`):
            The name of the Hammer model to use.
        client (`OpenAI`):
            An OpenAI client instance for making API calls.
        task_instruction (`str`):
            Instructions defining the task for the model.
        format_instruction (`str`):
            Instructions on how to format the output.

    Methods:
        from_config(`HammerConfig`):
            Class method to create an instance from an HammerConfig object.
        completion(`List[Dict[str, str]]`, `Optional[List[Dict[str, Any]]]`, `**kwargs`):
            Generate a chat completion based on provided messages and tools.
    """

    def __init__(
        self, 
        base_url: str,
        model: str,
        task_instruction: str=HammerConfig.TASK_INSTRUCTION, 
        format_instruction: str=HammerConfig.FORMAT_INSTRUCTION
    ):
        self.model_name = model
        self.client = OpenAI(base_url=base_url, api_key="EMPTY")
        self.task_instruction = task_instruction
        self.format_instruction = format_instruction
    
    @classmethod
    def from_config(cls, config: HammerConfig):
        return cls(
            model=config.MODEL_NAME,
            base_url=config.BASE_URL,
            task_instruction=config.TASK_INSTRUCTION,
            format_instruction=config.FORMAT_INSTRUCTION
        )
    
    def completion(self, messages, tools=None,temperature=0.0001, **kwargs):
        # Convert OpenAI-style functions to xLAM format
        if messages[0]['role'] == 'system':
            system_message = messages[0]
            messages = messages[1:]
        else:
            system_message = None
        try:    
            format_tools = self.convert_to_format_tool(tools) if tools else []
        except:
            format_tools = tools if tools else []
        
        user_query = ""

        for message in messages:
            user_query += f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"

        content = f"[BEGIN OF TASK INSTRUCTION]\n{self.task_instruction}\n[END OF TASK INSTRUCTION]\n\n"
        content += (
            "[BEGIN OF AVAILABLE TOOLS]\n"
            + json.dumps(tools)
            + "\n[END OF AVAILABLE TOOLS]\n\n"
        )
        content += f"[BEGIN OF FORMAT INSTRUCTION]\n{self.format_instruction}\n[END OF FORMAT INSTRUCTION]\n\n"
        formatted_prompt =  f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{content}<|im_end|>\n{user_query}<|im_start|>assistant\n"
        api_response = self.client.completions.create(
                model=self.model_name,
                temperature=temperature,
                prompt=formatted_prompt,
                max_tokens=32768
            )
        
        return api_response.choices[0].text

    def convert_to_format_tool(tools):

        if isinstance(tools, dict):
            format_tools = {
                "name": tools["name"],
                "description": tools["description"],
                "parameters": tools["parameters"].get("properties", {}),
            }

            for param in format_tools["parameters"].keys():
                if "properties" in format_tools["parameters"][param] and isinstance(
                    format_tools["parameters"][param]["properties"], dict
                ):
                    required = format_tools["parameters"][param].get("required", [])
                    format_tools["parameters"][param] = format_tools["parameters"][param]["properties"]
                    for p in required:
                        format_tools["parameters"][param][p]["required"] = True

            required = tools["parameters"].get("required", [])
            for param in required:
                format_tools["parameters"][param]["required"] = True
            for param in format_tools["parameters"].keys():
                if "default" in format_tools["parameters"][param]:
                    default = format_tools["parameters"][param]["default"]
                    format_tools["parameters"][param][
                        "description"
                    ] += f"default is '{default}'"
            return format_tools
        elif isinstance(tools, list):
            return [convert_to_format_tool(tool) for tool in tools]
        else:
            return tools

