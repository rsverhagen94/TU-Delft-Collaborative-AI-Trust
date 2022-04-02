import enum
import json
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


class LiarAgent(BaseAgent):

    def _sendMessage(self, mssg, sender):
        """
        Enable sending messages in one line of code
        """
        doors = [door['room_name'] for door in self._status.values()
                 if 'class_inheritance' in door
                 and 'Door' in door['class_inheritance']]
        blocks = self._world_state['goals']
        randommessage = random.choice([Messages.MOVING_TO_ROOM,
                                       Messages.OPENING_DOOR,
                                       Messages.SEARCHING_THROUGH,
                                       Messages.FOUND_GOAL_BLOCK1,
                                       Messages.PICKING_UP_GOAL_BLOCK2
                                       ])
        worldSize = self.state['World']['grid_shape']
        x = random.choice(range(1,worldSize[0]))
        y = random.choice(range(1,worldSize[1]))
        location = (x, y)
        message = ''
        if Messages.MOVING_TO_ROOM == randommessage:
            message = Messages.MOVING_TO_ROOM.value[0] + random.choice(doors)
        elif Messages.OPENING_DOOR == randommessage:
            message = Messages.OPENING_DOOR.value[0] + random.choice(doors)
        elif Messages.SEARCHING_THROUGH == randommessage:
            message = Messages.SEARCHING_THROUGH.value[0] + random.choice(doors)
        elif Messages.FOUND_GOAL_BLOCK1 == randommessage:
            block = random.choice(blocks)
            message = Messages.FOUND_GOAL_BLOCK1.value[0] + json.dumps(block['visualization']) + \
                      Messages.FOUND_GOAL_BLOCK2.value[0] + str(location)
        elif Messages.PICKING_UP_GOAL_BLOCK1 == randommessage:
            block = random.choice(blocks)
            message = Messages.PICKING_UP_GOAL_BLOCK1.value[0] + json.dumps(block['visualization']) + \
                      Messages.PICKING_UP_GOAL_BLOCK2.value[0] + str(location)

        if random.random() > 0.2:
            mssg = message

        super()._sendMessage(mssg, sender)
