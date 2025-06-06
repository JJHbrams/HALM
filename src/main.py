import os, sys
import json
from core_module.llm_agnet import LLMAgent

ROOT = os.getcwd()


def main(config):
    llm_model = LLMAgent(config)
    # reponse = llm_model.generate_response("Oh hi?")
    # print(reponse)
    while True:
        print("Query: ", end="", flush=True)
        input_buffer = sys.stdin.readline().strip()
        response = llm_model.generate_response(input_buffer)
        print(f"Answer:{response}")
        print("=" * 10, end="\n")

        # TTS
        llm_model.tts.speak(response)


if __name__ == "__main__":
    with open(f"{ROOT}/config/config.json", encoding="utf-8") as f:
        config = json.load(f)

    config["path"]["root"] = ROOT
    config["path"]["config"] = f"{ROOT}/config"
    config["path"]["logs"] = f"{ROOT}/logs"

    main(config)
