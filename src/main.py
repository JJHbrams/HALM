import os, sys
import json
from core_module.llm_agnet import LLMAgent

ROOT = os.getcwd()


def main(config):
    llm_model = LLMAgent(config)
    reponse = llm_model.generate_response(
        # "How to put the elephant in the refrigerator?"
        # "Do you need upgrading GPU?"
        "Oh hi?"
        # "Can you use korean profient?"
        # "내일 날씨좀 알려줘"
    )
    print(reponse)


if __name__ == "__main__":
    with open(f"{ROOT}/config/config.json") as f:
        config = json.load(f)
    main(config)
