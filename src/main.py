import os, sys
import json
from core_module.llm_agent import LLMAgent

ROOT = os.getcwd()


def main(config):
    llm_model = LLMAgent(config)
    # reponse = llm_model.generate_response("Oh hi?")
    # print(reponse)
    while True:
        print("\n\033[96mQuery: \033[0m", end="", flush=True)  # 밝은 하늘색
        input_buffer = sys.stdin.readline().strip()
        response = llm_model.generate_response(input_buffer)
        print(f"\033[92mAnswer:\033[0m{response}")  # 연두색
        print("\033[90m" + "=" * 50 + "\033[0m", end="\n")  # 회색 구분선

        # TTS
        llm_model.tts.speak(response)


if __name__ == "__main__":
    with open(f"{ROOT}/config/config.json", encoding="utf-8") as f:
        config = json.load(f)

    config["path"]["ROOT"] = ROOT
    config["path"]["CONFIG"] = f"{ROOT}/config"
    config["path"]["LOGS"] = f"{ROOT}/logs"

    main(config)
