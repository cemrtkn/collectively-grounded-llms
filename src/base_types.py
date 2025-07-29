from typing import Optional, Dict, List, Tuple, Union
from pydantic import BaseModel, Field


class Event(BaseModel):
    name: str = Field(..., description="Name of the event")
    event_idx: int
    game_idx: Union[int, str]
    exp_game_id: Optional[str] = None
    round_number: Optional[int] = None
    player_id: Optional[int] = None
    player_name: Optional[str] = None
    value: Optional[str] = None
    values: Optional[Dict[str, str]] = {}
    reward: Optional[float] = None
    disc_return: Optional[float] = None
    return_prediction: Optional[float] = None
    logp: Optional[float] = None


class Conversation(BaseModel):
    npc_name: Optional[str] = None
    npc_persona:Optional[str] = None
    player_name:Optional[str] = None
    player_persona: Optional[str] = None
    conversation: Optional[str] = None


class Game(BaseModel):
    game_idx: int
    exp_game_id: Optional[str] = None
    permutation_idx: Optional[int] = None
    n_rounds: int
    n_players: int
    discount_factor: float


class GameRound(BaseModel):
    round_number: int