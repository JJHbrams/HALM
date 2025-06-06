import os
import json
from datetime import datetime
from langchain_ollama import OllamaLLM

from voice_module.tts_agent import TTSAgent


class LLMAgent:
    def __init__(self, config):
        # configuration
        self.config = config
        self.cfg_personality = config["LLM"]["personality"]
        self.cfg_lang = config["LLM"]["language"]
        self.cfg_rule = config["LLM"]["rule"]
        self.cfg_attd = config["LLM"]["attitude"]
        self.cfg_example = config["LLM"]["example"]

        # Model 설정
        self.model = OllamaLLM(model=self.config["LLM"]["model"])

        # TTS 설정
        # self.tts = TTSAgent(config)

        # system prompt
        self.system_though = (
            f"Your name is {self.cfg_personality['name']}. "
            "You are a thinking module analyzing user input, making rational decisions."
            "Keep 'RULES'. "
            "'RULES' are rules for how you answer, You must follow it. "
            f"RULES: {self.cfg_rule}\n"
        )
        self.system_speech = (
            f"Your name is {self.cfg_personality['name']}, your personality is {self.cfg_personality['identity']}. "
            f"Answer about user input in {self.cfg_lang} language. "
            "Keep 'RULES'. "
            "'ATTIDUDES' are rules for how you answer, You must follow it. "
            "Refer 'EXAMPLES' when given query needs that format. "
            f"ATTIDUDES: {self.cfg_attd}\n"
            f"EXAMPLES: {self.cfg_example}\n"
        )

        # History 관리
        self.thought_history = []  # 생각 기록 (생각 모듈용)
        self.chat_history = []  # (user, assistant) 쌍 저장
        self.summary_history = []  # 요약 기록

        self.history_backup_path = os.path.join(
            config["path"]["logs"], "conversation", "chat_history.json"
        )
        os.makedirs(os.path.dirname(self.history_backup_path), exist_ok=True)
        # chat_history 복구 (최적화: 파일이 작을 때만, 예외 최소화)
        if os.path.exists(self.history_backup_path):
            try:
                with open(self.history_backup_path, "r", encoding="utf-8") as f:
                    self.chat_history = json.load(f)
            except Exception:
                self.chat_history = []

        self.summary_backup_path = os.path.join(
            config["path"]["logs"], "conversation", "summary_history.json"
        )
        os.makedirs(os.path.dirname(self.summary_backup_path), exist_ok=True)
        if os.path.exists(self.summary_backup_path):
            try:
                with open(self.summary_backup_path, "r", encoding="utf-8") as f:
                    self.summary_history = json.load(f)
            except Exception:
                self.summary_history = []

        self.think_backup_path = os.path.join(
            config["path"]["logs"], "conversation", "thought_history.json"
        )
        os.makedirs(os.path.dirname(self.think_backup_path), exist_ok=True)
        if os.path.exists(self.think_backup_path):
            try:
                with open(self.think_backup_path, "r", encoding="utf-8") as f:
                    self.thought_history = json.load(f)
            except Exception:
                self.thought_history = []

    def backup_history(self):
        # 최적화: 변경이 있을 때만 저장
        if (
            not self.thought_history
            and not self.chat_history
            and not self.summary_history
        ):
            return
        try:
            with open(self.think_backup_path, "w", encoding="utf-8") as f:
                json.dump(self.thought_history, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        try:
            with open(self.history_backup_path, "w", encoding="utf-8") as f:
                json.dump(self.chat_history, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        try:
            with open(self.summary_backup_path, "w", encoding="utf-8") as f:
                json.dump(self.summary_history, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def generate_thinking(self, user_input):
        # thought 모듈의 응답을 바탕으로 대화 응답 생성
        if self.summary_history and self.thought_history:
            summary_str = "\n".join(self.summary_history[-5:])
            # thought_str = "\n".join(self.thought_history[-5:])
            thought_str = "\n".join(
                [
                    f"(query):{item['query']}, (though):{item['thought']}"
                    for item in self.thought_history[-5:]
                ]
            )
            prompt = f"{self.system_though}\n(Recent Summaries):\n{summary_str}\n(Recent Thoughts):{thought_str}\nquery:{user_input}"
        else:
            prompt = f"{self.system_though}\nquery: {user_input}"

        response = self.model.invoke(prompt)
        return response

    def generate_answer(self, user_input, thought):
        # thought 모듈의 응답을 바탕으로 대화 응답 생성
        prompt = f"{self.system_speech}\nHere's query:{user_input}\n And your thought:{thought}\n"

        response = self.model.invoke(prompt)
        return response

    def generate_summary(self, user_input, answer):
        # 2차: 최근 대화기록과 누적 summary를 종합적으로 정리
        if self.chat_history:
            history_slice = self.chat_history[-10:]
            recent_history = "\n".join(
                f"User: {item['query']}\nAssistant: {item.get('answer', '')}"
                for item in history_slice
            )
            # 누적 summary(마지막 5개)
            summaries = [
                item.get("summary", "")
                for item in self.chat_history
                if item.get("summary")
            ]
            summary_slice = summaries[-5:]
            summary_str = "\n".join(summary_slice)
        else:
            recent_history = ""
            summary_str = ""
        summary_prompt = (
            "You are a conversation summarizer within 200 words.\n"
            "Summarize the conversation history and accumulate them with given summaries.\n"
            "Accumulated summury should contain long memories as possible.\n"
            f"Recent Conversation: {recent_history}\nQuery: {user_input}\nAnswer: {answer}\n"
            f"Accumulated Summaries: {summary_str}"
        )
        return self.model.invoke(summary_prompt)

    def generate_response(self, user_input):
        # thought 모듈용 응답 생성
        thought = self.generate_thinking(user_input)
        # 응답 생성
        answer = self.generate_answer(user_input, thought)
        # 요약 생성
        response = self.generate_summary(user_input, answer)

        # 생각 기록 저장
        now = datetime.now().replace(microsecond=0).isoformat()
        self.thought_history.append(
            {
                "timestamp": now,
                "query": user_input,
                "thought": thought,
            }
        )
        # 대화 히스토리 및 백업 (최적화: answer 필드에 1차, 2차 응답 분리)
        self.chat_history.append(
            {
                "timestamp": now,
                "query": user_input,
                "answer": answer,
            }
        )
        self.summary_history.append(f"({now}) {response}")
        self.backup_history()
        return answer
