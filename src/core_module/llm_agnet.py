import json
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate


def linearize(str_lst):
    res = "\n".join(str_lst)
    return res


class LLMAgent:
    def __init__(self, config):
        # configuration
        self.config = config
        self.cfg_personality = config["LLM"]["personality"]
        self.cfg_lang = config["LLM"]["language"]
        # self.cfg_rule = linearize(config["LLM"]["rule"])
        # self.cfg_example = linearize(config["LLM"]["example"])
        self.cfg_rule = config["LLM"]["rule"]
        self.cfg_example = config["LLM"]["example"]

        # model definition
        self.model = OllamaLLM(model=self.config["LLM"]["model"])

        # Prompt format
        self.prompt_template = PromptTemplate(
            input_variables=[
                "name",
                "identity",
                "query",
                "language",
                "rule",
                "example",
            ],
            template="Your name is {name}, your personality is {identity}. "
            "Answer about user input in {language}."
            "Keep 'RULES'"
            "Refer 'EXAMPLES' when given query needs that format"
            "'RULES' are rules for how you answer, You must follow it."
            "'EXAMPLES' show formats of your response."
            "RULES:{rule}"
            "EXAMPLES:{example}"
            "query: {query}",
        )

    def generate_response(self, user_input):
        # Generate a response using the LLM

        prompt = self.prompt_template.format(
            name=self.cfg_personality["name"],
            identity=self.cfg_personality["identity"],
            query=user_input,
            language=self.cfg_lang,
            rule=self.cfg_rule,
            example=self.cfg_example,
        )
        response = self.model.invoke(prompt)
        return response
