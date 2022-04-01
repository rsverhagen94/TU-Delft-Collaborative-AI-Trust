import json
from typing import final, List, Dict, Final
import enum, random

from bw4t.BW4TBrain import BW4TBrain
from matrx.agents.agent_utils.state import State
from matrx.agents.agent_utils.navigator import Navigator
from matrx.agents.agent_utils.state_tracker import StateTracker
from matrx.actions.door_actions import OpenDoorAction
from matrx.actions.object_actions import GrabObject, DropObject
from matrx.messages.message import Message


class Phase(enum.Enum):
    PLAN_PATH_TO_CLOSED_DOOR = 1,
    FOLLOW_PATH_TO_CLOSED_DOOR = 2,
    OPEN_DOOR = 3


def findRoom(location, state):
    room = None
    for item in state:
        if item.name.split('_')[0] == 'room' and item.location == location:
            room = item.room_name.split('_')[1]

    return room


class BaseLineAgent(BW4TBrain):

    def __init__(self, settings: Dict[str, object]):
        super().__init__(settings)
        self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR
        self._teamMembers = []
        self._trustBeliefs = {}
        self._teamStatus = {}
        self._teamObservedStatus = {}
        self._age = 0

    def initialize(self):
        super().initialize()
        self._state_tracker = StateTracker(agent_id=self.agent_id)
        self._navigator = Navigator(agent_id=self.agent_id,
                                    action_set=self.action_set, algorithm=Navigator.A_STAR_ALGORITHM)

    def filter_bw4t_observations(self, state):
        agent_name = state[self.agent_id]['obj_id']

        if len(self._teamMembers) == 0:
            for member in state['World']['team_members']:
                if member != agent_name and member not in self._teamMembers:
                    self._teamMembers.append(member)
                    self._trustBeliefs[member] = {'rating': 0.5, 'age': self._age}

        for item in state:
            if item.isAgent:
                self._teamObservedStatus[item.name] = {'location': item.location, 'is_carrying': item.is_carrying,
                                                       'age': self._age}
                self._sendMessage('status of ' + item.name + ': location is '
                                  + item.location + 'and is carrying ' + item.is_carrying, agent_name)

        receivedMessages = self._processMessages(self._teamMembers)

        for member in self._teamMembers:
            if self._teamObservedStatus[member].age >= 5:
                self._teamObservedStatus[member] = None

        for member in self._teamMembers:
            for message in receivedMessages[member]:
                self._parseMessage(message.content, member)

        # Update trust beliefs for team members
        self._trustBlief(agent_name, state)
        return state

    def decide_on_bw4t_action(self, state: State):
        self._age += 1
        agent_name = state[self.agent_id]['obj_id']
        # Add team members

        while True:
            if Phase.PLAN_PATH_TO_CLOSED_DOOR == self._phase:
                self._navigator.reset_full()
                closedDoors = [door for door in state.values()
                               if 'class_inheritance' in door and 'Door' in door['class_inheritance'] and not door[
                        'is_open']]
                if len(closedDoors) == 0:
                    return None, {}
                # Randomly pick a closed door
                self._door = random.choice(closedDoors)
                doorLoc = self._door['location']
                # Location in front of door is south from door
                doorLoc = doorLoc[0], doorLoc[1] + 1
                # Send message of current action
                self._sendMessage('Moving to door of ' + self._door['room_name'], agent_name)
                self._navigator.add_waypoints([doorLoc])
                self._phase = Phase.FOLLOW_PATH_TO_CLOSED_DOOR

            if Phase.FOLLOW_PATH_TO_CLOSED_DOOR == self._phase:
                self._state_tracker.update(state)
                # Follow path to door
                action = self._navigator.get_move_action(self._state_tracker)
                if action is not None:
                    return action, {}
                self._phase = Phase.OPEN_DOOR

            if Phase.OPEN_DOOR == self._phase:
                self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR
                # Open door
                return OpenDoorAction.__name__, {'object_id': self._door['obj_id']}

    def _sendMessage(self, mssg, sender):
        '''
        Enable sending messages in one line of code
        '''
        msg = Message(content=mssg, from_id=sender)
        if msg.content not in self.received_messages:
            self.send_message(msg)

    def _processMessages(self, teamMembers):
        '''
        Process incoming messages and create a dictionary with received messages from each team member.
        '''
        receivedMessages = {}
        for member in teamMembers:
            receivedMessages[member] = []
        for mssg in self.received_messages:
            for member in teamMembers:
                if mssg.from_id == member:
                    receivedMessages[member].append(mssg.content)
        return receivedMessages

    def _trustBlief(self, name, state):
        '''
        Baseline implementation of a trust belief. Creates a dictionary with trust belief scores for each team member, for example based on the received messages.
        '''

        # You can change the default value to your preference

        for member in self._teamMembers:
            if self._teamStatus[member].action == 'searching':
                if self._teamObservedStatus[member] is not None:
                    if findRoom(self._teamObservedStatus[member].location, state) != findRoom(
                            self._teamStatus[member].location, state):
                        self._trustBeliefs[member] -= 0.1 * 1 / self._age
                    else:
                        self._trustBeliefs[member] += 0.1 * 1 / self._age
            if self._teamStatus[member].action == 'carrying':
                if self._teamObservedStatus[member] is not None:
                    if self._teamObservedStatus[member].is_carrying != self._teamStatus[member].block:
                        self._trustBeliefs[member] -= 0.1 * 1 / self._age
                    else:
                        self._trustBeliefs[member] += 0.1 * 1 / self._age

    def _parseMessage(self, message, member):
        string_list = message.split(" ")
        if string_list[0] == "Opening" and string_list[1] == "door":
            room_number = string_list[3].split("_")[1]
            self._teamStatus[member] = {'action': 'opening', 'room': room_number, 'age': self._age}
        if string_list[0] == "Searching" and string_list[1] == "through":
            room_number = string_list[2].split("_")[1]
            self._teamStatus[member] = {'action': 'searching', 'room': room_number, 'age': self._age}
        if string_list[0] == "Found" and string_list[1] == "goal":
            block = message.split('{')[1]
            block = '{' + block.split('}')[0] + '}'
            block = json.loads(block)
            self._teamStatus[member] = {'action': 'finding', 'block': block, 'age': self._age}
        if string_list[0] == "Picking" and string_list[1] == "up":
            block = message.split('{')[1]
            block = '{' + block.split('}')[0] + '}'
            block = json.loads(block)
            self._teamStatus[member] = {'action': 'carrying', 'block': block, 'age': self._age}
        if string_list[0] == "Dropping" and string_list[1] == "goal":
            block = message.split('{')[1]
            block = '{' + block.split('}')[0] + '}'
            block = json.loads(block)
            self._teamStatus[member] = {'action': 'dropping', 'block': block, 'age': self._age}
