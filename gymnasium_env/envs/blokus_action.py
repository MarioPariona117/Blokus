from typing import Tuple

class Action():

    @staticmethod
    def id_to_tuple(action_id: int) -> Tuple[int, int, int, int]:

    @staticmethod
    def tuple_to_id(action_tuple: Tuple[int, int, int, int]) -> int:
        
    def __init__(self, action_id: int | None = None, action_tuple: Tuple[int, int, int, int] | None = None):
        if action_id is not None:
            self.action_id = action_id
            self.action_tuple = None
        elif action_tuple is not None:
            self.action_id = None
            self.action_tuple = action_tuple
        else:
            raise ValueError("Either action_id or action_tuple must be provided")
        
    