import uuid
from typing import List, Dict
from datetime import datetime


class Message:
    def __init__(self, role: str, content: str = None, uniq_id: str = None):
        self.uniq_id = uniq_id if uniq_id is not None else str(uuid.uuid4())
        self.role = role
        self.content = content


class Dialogue:
    def __init__(self):
        self.dialogue: List[Message] = []
        # 获取当前时间
        self.current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    def put(self, message: Message):
        self.dialogue.append(message)

    def get_llm_dialogue(self) -> List[Dict[str, str]]:
        dialogue = []
        for m in self.dialogue:
            dialogue.append({"role": m.role, "content": m.content})
        return dialogue

def chat_history_to_dialogue(chat_history: list[dict[str, str]]) -> Dialogue:
    """
    将 chat history（列表形式，每个元素为包含 'role' 和 'content' 的字典）
    转换为 Dialogue 对象。
    """
    dialogue = Dialogue()
    for entry in chat_history:        
        message = Message(role=entry["role"], content=entry["content"])
        dialogue.put(message)
    return dialogue

