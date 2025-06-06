import os
import json
from datetime import datetime
from langchain_ollama import OllamaLLM

from voice_module.tts_agent import TTSAgent


def linearize(str_lst):
    res = "\n".join(str_lst)
    return res


class LLMAgent:
    def __init__(self, config):
        # configuration
        self.config = config
        self.cfg_personality = config["LLM"]["personality"]
        self.cfg_lang = config["LLM"]["language"]
        self.cfg_rule = config["LLM"]["rule"]
        self.cfg_example = config["LLM"]["example"]

        # TTS 설정
        self.tts = TTSAgent()

        # system prompt
        self.system_prompt = (
            f"Your name is {self.cfg_personality['name']}, your personality is {self.cfg_personality['identity']}. "
            f"Answer about user input in {self.cfg_lang}. "
            "Keep 'RULES'. "
            "'RULES' are rules for how you answer, You must follow it. "
            "Refer 'EXAMPLES' when given query needs that format. "
            "'EXAMPLES' show formats of your response. "
            f"RULES: {self.cfg_rule}\n"
            f"EXAMPLES: {self.cfg_example}\n"
        )
        self.model = OllamaLLM(model=self.config["LLM"]["model"])
        self.prompt_template = None  # PromptTemplate 미사용, 제거

        self.chat_history = []  # (user, assistant) 쌍 저장
        self.history_backup_path = os.path.join(
            config["path"]["LOGS"], "conversation", "chat_history.json"
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
            config["path"]["LOGS"], "conversation", "summary_history.json"
        )
        os.makedirs(os.path.dirname(self.summary_backup_path), exist_ok=True)
        self.summary_history = []
        if os.path.exists(self.summary_backup_path):
            try:
                with open(self.summary_backup_path, "r", encoding="utf-8") as f:
                    self.summary_history = json.load(f)
            except Exception:
                self.summary_history = []

    def backup_history(self):
        # 최적화: 변경이 있을 때만 저장
        if not self.chat_history and not self.summary_history:
            return
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

    def generate_response_1st(self, user_input):
        # 1차: summary도 참조하여 자유 응답 생성
        if self.summary_history:
            summary_str = "\n".join(self.summary_history[-5:])
            prompt = f"{self.system_prompt}\n(Recent Summaries):\n{summary_str}\nquery: {user_input}"
        else:
            prompt = f"{self.system_prompt}\nquery: {user_input}"
        response = self.model.invoke(prompt)
        return response

    def generate_response_2nd(self, user_input, answer):
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
            "You are a conversation summarizer.\n"
            "Summarize the following conversation history and accumulated summaries in a concise and informative way.\n"
            f"Recent Conversation: {recent_history}\nUser: {user_input}\nAssistant: {answer}\n"
            f"Accumulated Summaries: {summary_str}"
        )
        return self.model.invoke(summary_prompt)

    def generate_response(self, user_input):
        # 1차 응답 생성
        answer = self.generate_response_1st(user_input)
        # 2차 포맷팅
        response = self.generate_response_2nd(user_input, answer)
        # 대화 히스토리 및 백업 (최적화: answer 필드에 1차, 2차 응답 분리)
        now = datetime.now().replace(microsecond=0).isoformat()
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
