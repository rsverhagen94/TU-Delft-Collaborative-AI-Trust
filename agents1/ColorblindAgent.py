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
    OPEN_DOOR = 3,
    ENTER_ROOM = 4,
    TRAVERSE_ROOM =5,
    DELIVER_ITEM =6


class ColorblindAgent(BW4TBrain):

    def __init__(self, settings: Dict[str, object]):
        super().__init__(settings)
        self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR
        self._teamMembers = []
        self.desired_objects = []

    def initialize(self):
        super().initialize()
        self._state_tracker = StateTracker(agent_id=self.agent_id)
        self._navigator = Navigator(agent_id=self.agent_id,
                                    action_set=self.action_set, algorithm=Navigator.A_STAR_ALGORITHM)

    def filter_bw4t_observations(self, state):
        return state

    def decide_on_bw4t_action(self, state: State):
        agent_name = state[self.agent_id]['obj_id']
        # Add team members
        for member in state['World']['team_members']:
            if member != agent_name and member not in self._teamMembers:
                self._teamMembers.append(member)
                # Process messages from team members
        receivedMessages = self._processMessages(self._teamMembers)
        # Update trust beliefs for team members
        self._trustBlief(self._teamMembers, receivedMessages)

        self.desired_objects = list(map(
                    lambda x: x["visualization"],[wall for wall in state.values() if 'class_inheritance' in wall and 'GhostBlock' in wall['class_inheritance']]))
        #print(self.desired_objects)

        while True:
            if Phase.ENTER_ROOM == self._phase:
                #
                # print("PICK-----------------------------------------")
                room = self._door['room_name']

                area = list(map(
                    lambda x: x["location"],
                    [wall for wall in state.get_room_objects(room)
                     if 'class_inheritance' in wall and 'AreaTile' in wall['class_inheritance'] and
                     "is_drop_zone" not in wall]))

                # wall_locations = list(map(
                #     lambda x: x["location"],
                #     [wall for wall in state.get_room_objects(room)
                #                   if 'class_inheritance' in wall and 'Wall' in wall['class_inheritance']]))

                sorted_by_xy = sorted(sorted(area, key=lambda x: x[1]))
                # AreaTile

                #self._traverseRoom(min_xy, max_xy)

                self._navigator.reset_full()
                self._navigator.add_waypoints(sorted_by_xy)

                self._phase = Phase.TRAVERSE_ROOM

            if Phase.TRAVERSE_ROOM == self._phase:
                self._state_tracker.update(state)

                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    print(
                    list(map(
                        lambda x: x["visualization"], [wall for wall in state.get_closest_with_property("is_collectable")])))


                    return action, {}

                self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR
                print("AAA")

                #return GrabObject.__name__, {'object_id': self._door['obj_id']}


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
                if action != None:
                    return action, {}
                self._phase = Phase.OPEN_DOOR

            if Phase.OPEN_DOOR == self._phase:
                self._phase = Phase.ENTER_ROOM
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

    def _trustBlief(self, member, received):
        '''
        Baseline implementation of a trust belief. Creates a dictionary with trust belief scores for each team member, for example based on the received messages.
        '''
        # You can change the default value to your preference
        default = 0.5
        trustBeliefs = {}
        for member in received.keys():
            trustBeliefs[member] = default
        for member in received.keys():
            for message in received[member]:
                if 'Found' in message and 'colour' not in message:
                    trustBeliefs[member] -= 0.1
                    break
        return trustBeliefs

    def _traverseRoom(self, min_xy, max_xy):
        self._navigator.reset_full()

        list_coordinates = []
        for x in range(min_xy[0]+1, max_xy[0]):
            for y in range(min_xy[1]+1, max_xy[1]-1):
                list_coordinates.append((x, y))
                print(x, y)

        self._navigator.add_waypoints(list_coordinates)
