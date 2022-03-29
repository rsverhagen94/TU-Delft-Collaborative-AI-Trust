import enum
import random

from matrx.agents.agent_utils.state import State

from .Team36BaseAgent import BaseAgent
from typing import Dict


class Messages(enum.Enum):
    MOVING_TO_ROOM = "Moving to ",
    OPENING_DOOR = "Opening door of ",
    SEARCHING_THROUGH = "Searching through ",
    FOUND_GOAL_BLOCK1 = "Found goal block ",
    FOUND_GOAL_BLOCK2 = " at location ",
    PICKING_UP_GOAL_BLOCK1 = "Picking up goal block ",
    PICKING_UP_GOAL_BLOCK2 = " at location ",
    DROPPED_GOAL_BLOCK1 = "Dropped goal block ",
    DROPPED_GOAL_BLOCK2 = " at drop location ",


class LazyAgent(BaseAgent):

    def __init__(self, settings: Dict[str, object]):
        super().__init__(settings)
        self.__state = None

    def initialize(self):
        super().initialize()
        self.__state = None

    def decide_on_bw4t_action(self, state: State):
        self.__state = state
        super().decide_on_bw4t_action(state)
        
    def _sendMessage(self, mssg, sender):
        """
        Enable sending messages in one line of code
        """
        doors = [door['room_name'] for door in self.__state.values()
                 if 'class_inheritance' in door
                 and 'Door' in door['class_inheritance']]
        blocks = self._world_state['goals']
        randommessage = random.choice([Messages.MOVING_TO_ROOM,
                                       Messages.OPENING_DOOR,
                                       Messages.SEARCHING_THROUGH,
                                       Messages.FOUND_GOAL_BLOCK1,
                                       Messages.PICKING_UP_GOAL_BLOCK2,
                                       Messages.DROPPED_GOAL_BLOCK1])
        message = ''
        if Messages.MOVING_TO_ROOM == randommessage:
            message = Messages.MOVING_TO_ROOM.value[0] + random.choice(doors)
        elif Messages.OPENING_DOOR == randommessage:
            message = Messages.OPENING_DOOR.value[0] + random.choice(doors)
        elif Messages.SEARCHING_THROUGH == randommessage:
            message = Messages.SEARCHING_THROUGH.value[0] + random.choice(doors)
        elif Messages.FOUND_GOAL_BLOCK1 == randommessage:
            block = random.choice(blocks)
            message = Messages.FOUND_GOAL_BLOCK1.value[0] + repr(block['visualization']) + \
                      Messages.FOUND_GOAL_BLOCK2.value[0] + repr(block['location'])
        elif Messages.PICKING_UP_GOAL_BLOCK1 == randommessage:
            block = random.choice(blocks)
            message = Messages.PICKING_UP_GOAL_BLOCK1.value[0] + repr(block['visualization']) + \
                      Messages.PICKING_UP_GOAL_BLOCK2.value[0] + repr(block['location'])
        elif Messages.DROPPED_GOAL_BLOCK1 == randommessage:
            block = random.choice(blocks)
            message = Messages.DROPPED_GOAL_BLOCK1.value[0] + repr(block['visualization']) + \
                      Messages.DROPPED_GOAL_BLOCK2.value[0] + repr(block['location'])

        if random.random() > 0.2:
            mssg = message

        super()._sendMessage(mssg, sender)
