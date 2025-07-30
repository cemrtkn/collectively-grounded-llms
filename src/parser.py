from typing import Dict, List, Optional, Any
import yaml
from pydantic import BaseModel, Field, root_validator

from src.base_types import Event, Conversation


class EventConfig(BaseModel):
    idx: int = Field(..., description="Index of the event")
    name: str = Field(..., description="Name of the event")
    text: str = Field(..., description="Unparsed text of event")
    allowed_values: Optional[List[str]] = Field(
        default=None, description="List of allowed values for the targets"
    )
    generation_config: Optional[dict] = Field(
        default=None, description="Generation config for the target chunk"
    )
    stop_token: Optional[str] = Field(default=None, description="String to stop generation")
    instruction: Optional[str] = Field(default=None, description="String for instructing gpts")
    eot_id_bias: Optional[float] = Field(default=None, description="Bias for end of text token")
    use_prefix_allowed_tokens_fn: Optional[bool] = Field(
        default=True, description="Whether to set prefix_allowed_tokens_fn"
    )

class ToyParserConfig(BaseModel):
    fields: dict


class RoundsToKeepParserConfig(BaseModel):
    early_rounds: Optional[int] = Field(
        default=None, description="Number of early rounds to keep in the history", ge=0
    )
    latest_rounds: Optional[int] = Field(
        default=None, description="Number of latest rounds to keep in the history", ge=0
    )


class ParserConfig(BaseModel):
    events: Dict[str, EventConfig] = Field(
        ..., description="Dictionary of event configs keyed by event name"
    )
    target_start_mark: str = Field(
        default='"', description="This marks the start of the target text"
    )
    target_stop_mark: str = Field(default='"', description="This marks the stop of the target text")
    rounds_to_keep: RoundsToKeepParserConfig = Field(
        default=None, description="Config for keeping only a subset of rounds into the history"
    )

    @root_validator(pre=True)
    def set_event_idx_and_name(cls, values):
        events = values.get("events")
        for idx, (name, event) in enumerate(events.items()):
            event["idx"] = idx
            # Set the name if not present
            event.setdefault("name", name)
        return values

    @classmethod
    def default(cls):
        with open("example_configs/parser.yml", "r") as file:
            return cls.parse_obj(yaml.safe_load(file)["parser_config"])

    @classmethod
    def load_from_file(cls, file_name):
        with open(file_name, "r") as file:
            return cls.parse_obj(yaml.safe_load(file)["parser_config"])


class EventParser:
    def __init__(self, config: ParserConfig) -> None:
        self.config = config
        self.event_configs = config.events
        self.rounds_to_keep = config.rounds_to_keep

    def sanitize_event_values(self, events: List[Event]):
        """Newlines or double quotes in event values break our code.
        This replaces newlines with spaces and double quotes with single quotes
        Also replaces empty values with a single space"""
        for event in events:
            if event.value is not None:
                event.value = str(event.value).replace('"', "'").replace("\n", " ")
                if len(event.value) == 0:  # All empty messages get turned into a single space
                    event.value = " "

    def value_parse(self, events: List[Event], auto_sanitize=True) -> list[str]:
        if auto_sanitize:
            self.sanitize_event_values(events)
        texts = []
        gi = None
        for i, e in enumerate(events):
            if e.game_idx != gi:
                texts.append([])
                gi = e.game_idx
            ec = self.event_configs[e.name]
            texts[-1].append(ec.text.format(**e.dict()))

        return ["\n".join(x) for x in texts]

    def parse(
        self,
        events: List[Event],
        up_to_target=False,
        auto_sanitize=True,
    ) -> List[str]:
        if auto_sanitize:
            self.sanitize_event_values(events)
        texts = []
        gi = None
        for i, e in enumerate(events):
            if e.game_idx != gi:
                texts.append(new_game_texts := [])
                gi = e.game_idx

            if (
                self.config.rounds_to_keep
                and self.config.rounds_to_keep.early_rounds
                and self.config.rounds_to_keep.latest_rounds
            ):
                assert (
                    self.rounds_to_keep.early_rounds is not None
                    and self.rounds_to_keep.latest_rounds is not None
                ), "early_rounds and latest_rounds must be set"
                current_round = max(
                    [
                        event.round_number
                        for event in events
                        if event.game_idx == gi and event.name in ["round_start", "game_start"]
                    ]
                )
                start_round = self.rounds_to_keep.early_rounds
                end_round = current_round - self.rounds_to_keep.latest_rounds - 1
                # include only a subset of rounds (or none) into the history
                if e.round_number >= start_round and e.round_number <= end_round:
                    # replace the omitted rounds with "..."
                    if e.name == "round_start" and e.round_number == start_round:
                        new_game_texts.append("\n...")
                    # skip the event if it's round is not in the list of rounds to keep
                    continue

            ec = self.event_configs[e.name]
            # print(ec.text, e)
            new_game_texts.append(ec.text.format(**e.dict()))

        # remove last item from new_game_texts and parse it again for the instruct model
        new_game_texts.pop()
        last_event = events[-1]

        ec = self.event_configs[f"{last_event.name}_prompt"]
        new_game_texts.append(ec.text.format(**last_event.dict()))

        ec = self.event_configs[f"{last_event.name}_ans"]
        ec.text = ec.text.split("{value}")[0] if up_to_target else ec.text
        new_game_texts.append(ec.text.format(**last_event.dict()))

        return ["\n".join(x) for x in texts]

    def sft_parse_per_rounds(
        self, events: List[Event], auto_sanitize=False, max_rounds=30, n_players=2
    ) -> Dict[str, List[Any]]:
        """
        Structure the game data in a Round-base format, where each data point of a round t will include:

        - [Optional] first few rounds: Gameplay from Round 0 to early_rounds
        - [Optional] latest few rounds: Gameplay from Round t-latest_rounds to Round t-1
        - Gameplay at Round t (current round)
            For each game and each player anf each round, we create many datapoints: depends on the number of events you want to generate
            Example: For CoopBot project we generate 2 datapoints for each round t:
                a. For message, the target is the current player message at round t without including the other player's message at round t
                b. For action, the target is the current player action at round t without including the other player's action at round t (we add to the history the messages at current round t)

        To include only the current round t (without any history):
            set early_rounds and latest_rounds to 0
        """

        game_dict = {}
        for e in events:
            if e.game_idx not in game_dict:
                game_dict[e.game_idx] = []
            game_dict[e.game_idx].append(e)

        texts = []
        exp_game_id_list = []
        round_number_list = []
        event_name_list = []
        player_id_list = []
        # for each event that has a prompt and an answer, we create a datapoint
        event_names_with_prompt_and_ans = [
            key
            for key in self.event_configs.keys()
            if key + "_prompt" in self.event_configs.keys()
            and key + "_ans" in self.event_configs.keys()
        ]
        for _, game_events in game_dict.items():
            for current_round in range(max_rounds):
                history_events = [
                    event for event in game_events if event.round_number < current_round
                ]
                current_round_events = [
                    event for event in game_events if event.round_number == current_round
                ]

                for event_name in event_names_with_prompt_and_ans:
                    for player_id in range(n_players):
                        history_events_copy = history_events.copy()
                        for event in current_round_events:
                            # Hack: the events names are sorted
                            if event.name == event_name and event.player_id != player_id:
                                # players send event_name simultaneously, therefore we don't want to include the other player's current event
                                continue
                            history_events_copy.append(event)

                            if event.name == event_name and event.player_id == player_id:
                                break
                            
                        last_event = history_events_copy[-1]
                        text = self.parse(
                            history_events_copy, auto_sanitize=False, up_to_target=False
                        )
                        texts.append(text[0])
                        exp_game_id_list.append(last_event.exp_game_id)
                        round_number_list.append(last_event.round_number)
                        event_name_list.append(last_event.name)
                        player_id_list.append(last_event.player_id)

        return {"exp_game_id": exp_game_id_list, "round_number": round_number_list, "event_name": event_name_list, "player_id": player_id_list, "text": texts}

class ToyParser():
    def __init__(self, config: ToyParserConfig) -> None:
        self.config = config
        self.field_configs = config.fields


    def parse(self, convs: List[Conversation]):
        """Formats an example dictionary into a model input string using parser_config."""
        text_results ={"text":[]}

        for conv in convs:
            parts = []
            for key in self.field_configs:
                text_template = self.field_configs[key]["text"]
                try:
                    # Format using keys in the example
                    filled_text = text_template.format(**conv.__dict__)
                except KeyError as e:
                    raise ValueError(f"Missing key {e} in example: {conv}")
                
                parts.append(filled_text)

            text_results["text"].append("\n".join(parts))

        return text_results

class ParseEventConfig(BaseModel):
    events_dataset: str = Field(default="data/dev", description="Path to the events dataset")
    config: str = Field(default="config/dev/dev.yml", description="Path to the config file")
