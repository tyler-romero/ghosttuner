import os
from openai import OpenAI
import json
from typing import Dict, Optional

SYSTEM_PROMPT = "You are a Machine Learning expert."


def get_initial_message(model_type: str, search_space: str, total_budget: int = 10) -> str:
    prompt = (
        f"You are helping tune hyperparameters for a {model_type}. "
        "Training is done with Sklearn. This is our hyperparameter search space:\n"
    )
    prompt += search_space
    prompt += (
        f"\nWe have a budget to try {total_budget} configurations in total. "
        "You will get the validation error rate (1 - accuracy) before you need to "
        "specify the next configuration. The goal is to find the configuration that "
        "minimizes the error rate with the given budget, so you should explore "
        "different parts of the search space if the loss is not changing. "
    )

    prompt += '\nProvide a config in JSON format. Do not put new lines or any extra characters in the response.\nExample config: {"C": x, "gamma": y}.\n Config:'
    return prompt


def get_transition_message(loss: float, use_cot: bool = False) -> str:
    if use_cot:
        prompt = f"""loss = {loss:.4 e}. Write two lines as follows:
Analysis: Up to a few sentences describing what worked so far and what to choose next
Config: (JSON config)"""
    else:
        prompt = f"loss = {loss:.4e}. Specify the next config, do not add anything else in your response.\nConfig:"

    return prompt


def recycle_response(response) -> Dict:
    return {"role": "assistant", "content": response.choices[0].message.content}


class ConversationManager:
    def __init__(
        self,
        model_type: str,
        search_space: str,
        budget: int = 10,
        use_cot: bool = False,
        model: str = "gpt-4-1106-preview",
    ):
        self.model_type = model_type
        self.search_space = search_space
        self.budget = budget
        self.use_cot = use_cot
        self.model = model
        self.turn = 0
        self.openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.history = [{"role": "system", "content": SYSTEM_PROMPT}]

    def sample_hyperparameters(self, loss: Optional[float] = None) -> Dict:
        if self.turn == 0:
            message = get_initial_message(self.model_type, self.search_space, self.budget)
        else:
            if loss is None:
                raise ValueError("After the first trial a loss must be provided.")
            message = get_transition_message(loss, use_cot=self.use_cot)

        self.history.append({"role": "user", "content": message})

        response = self.openai_client.chat.completions.create(
            messages=self.history,
            model=self.model,
            response_format={"type": "json_object"} if not self.use_cot else None,
            temperature=0.0,
        )
        print(response)
        self.history.append(recycle_response(response))

        dict_response = json.loads(response.choices[0].message.content)
        self.turn += 1
        return dict_response
