from pydantic import BaseModel

class Conversation(BaseModel):
    npc_name: str = None
    npc_persona:str = None
    player_name:str = None
    player_persona: str = None
    conversation: str = None