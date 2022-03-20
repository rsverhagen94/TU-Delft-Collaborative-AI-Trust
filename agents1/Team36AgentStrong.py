from typing import final, List, Dict, Final
import enum, random
from agents1.BW4TBaselineAgent import BaseLineAgent
from matrx.agents.agent_utils.state import State
from matrx.agents.agent_utils.navigator import Navigator
from matrx.agents.agent_utils.state_tracker import StateTracker
from matrx.actions.door_actions import OpenDoorAction
from matrx.actions.object_actions import GrabObject, DropObject
from matrx.messages.message import Message

class Phase(enum.Enum):
    PLAN_PATH_TO_CLOSED_DOOR=1,
    FOLLOW_PATH_TO_CLOSED_DOOR=2,
    OPEN_DOOR=3

class StrongAgent(BaseLineAgent):

    def __init__(self, settings:Dict[str,object]):
        super().__init__(settings)

    def initialize(self):
        super().initialize()

    def filter_bw4t_observations(self, state):
        return state

    def decide_on_bw4t_action(self, state:State):
        agent_name = state[self.agent_id]['obj_id']
        # Add team members
        for member in state['World']['team_members']:
            if member!=agent_name and member not in self._teamMembers:
                self._teamMembers.append(member)       
        # Process messages from team members
        receivedMessages = self._processMessages(self._teamMembers)
        # Update trust beliefs for team members
        self._trustBlief(self._teamMembers, receivedMessages)
        
        while True:
            if Phase.PLAN_PATH_TO_CLOSED_DOOR==self._phase:
                self._navigator.reset_full()
                closedDoors = [door for door in state.values()
                    if 'class_inheritance' in door and 'Door' in door['class_inheritance'] and not door['is_open']]
                if len(closedDoors)==0:
                    return None, {}
                # Randomly pick a closed door
                self._door = random.choice(closedDoors)
                doorLoc = self._door['location']
                # Location in front of door is south from door
                doorLoc = doorLoc[0],doorLoc[1]+1
                # Send message of current action
                self._sendMessage('Moving to door of ' + self._door['room_name'], agent_name)
                self._navigator.add_waypoints([doorLoc])
                self._phase=Phase.FOLLOW_PATH_TO_CLOSED_DOOR

            if Phase.FOLLOW_PATH_TO_CLOSED_DOOR==self._phase:
                self._state_tracker.update(state)
                # Follow path to door
                action = self._navigator.get_move_action(self._state_tracker)
                if action!=None:
                    return action, {}   
                self._phase=Phase.OPEN_DOOR

            if Phase.OPEN_DOOR==self._phase:
                self._phase=Phase.PLAN_PATH_TO_CLOSED_DOOR
                # Open door
                return OpenDoorAction.__name__, {'object_id':self._door['obj_id']}
    