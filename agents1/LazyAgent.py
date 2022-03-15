import enum
import random
from typing import Dict

from matrx.actions.door_actions import OpenDoorAction
from matrx.actions.object_actions import DropObject
from matrx.actions.object_actions import GrabObject
from matrx.agents.agent_utils.navigator import Navigator
from matrx.agents.agent_utils.state import State
from matrx.agents.agent_utils.state_tracker import StateTracker
from matrx.messages.message import Message
import re
import json

from bw4t.BW4TBrain import BW4TBrain


class Phase(enum.Enum):
    PLAN_PATH_TO_CLOSED_DOOR = 1,
    FOLLOW_PATH_TO_CLOSED_DOOR = 2,
    OPEN_DOOR = 3
    SEARCH_ROOM = 4
    MOVING_BLOCK = 5
    START = 6
    GRAB = 7
    MOVE_TO_OBJECT = 8
    STOP = 9


class LazyAgent(BW4TBrain):

    def __init__(self, settings: Dict[str, object]):
        super().__init__(settings)
        self._phase = Phase.START
        self._teamMembers = []

    def initialize(self):
        super().initialize()
        self._door = None
        self._state_tracker = StateTracker(agent_id=self.agent_id)
        self._navigator = Navigator(agent_id=self.agent_id,
                                    action_set=self.action_set, algorithm=Navigator.A_STAR_ALGORITHM)
        self._objects = []
        self._goal_objects_found = []
        self._goal_objects = []
        self._goal_object_delivered = []
        self._current_obj = None
        self._can_be_lazy = True

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
        for member in self._teamMembers:
            for msg in receivedMessages[member]:
                if "Found goal block " in msg:
                    #print(msg)
                    pattern = re.compile("{(.* ?)}")
                    vis = re.search(pattern, msg).group(0)
                    #vis = vis.replace("\'", "\"")
                    pattern2 = re.compile("\[(.* ?)\]")
                    loc = re.search(pattern2, msg).group(0)
                    print(vis, loc)

                    loc = json.loads(loc)
                    vis = json.loads(vis)

                    print(vis, loc)

        # Update trust beliefs for team members
        self._trustBlief(self._teamMembers, receivedMessages)

        while True:

            if self.__is_lazy() and self._phase != Phase.START and self._can_be_lazy and self._phase in [
                Phase.SEARCH_ROOM,
                Phase.MOVING_BLOCK,
                Phase.MOVE_TO_OBJECT]:
                self._can_be_lazy = False
                if state[agent_name]['is_carrying']:
                    # self._goal_objects_found.remove(self._current_obj)
                    obj = self._current_obj
                    obj['location'] = state[agent_name]['location']
                    self._goal_objects_found.append(obj)
                    self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR
                    return DropObject.__name__, {'object_id': obj['obj_id']}
            else:
                self._can_be_lazy = True

            if Phase.START == self._phase:
                # 'Drop_off_0', 'Drop_off_0_1', 'Drop_off_0_2'
                # {'drop_zone_nr': 0, 'is_drop_zone': True, 'is_goal_block': False, 'is_collectable': False, 'name': 'Drop off 0', 'obj_id': 'Drop_off_0', 'location': (12, 21), 'is_movable': False, 'carried_by': [], 'is_traversable': True,
                # Collect_Block', 'Collect_Block_1', 'Collect_Block_2'
                # {'drop_zone_nr': 0, 'is_drop_zone': False, 'is_goal_block': True, 'is_collectable': False, 'name': 'Collect Block', 'obj_id': 'Collect_Block', 'location': (12, 23), 'is_movable': False, 'carried_by': [], 'is_traversable': True

                self._goal_objects.append(state['Collect_Block'])
                self._goal_objects.append(state['Collect_Block_1'])
                self._goal_objects.append(state['Collect_Block_2'])

                self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR

            if Phase.PLAN_PATH_TO_CLOSED_DOOR == self._phase:
                self._current_obj = None
                self._navigator.reset_full()
                closedDoors = [door for door in state.values()
                               if 'class_inheritance' in door and 'Door' in door['class_inheritance'] and not door[
                        'is_open']]
                if len(closedDoors) == 0:
                    self._door = random.choice([door for door in state.values()
                                                if 'class_inheritance' in door and 'Door' in door['class_inheritance']])
                    if self._goal_objects_found:
                        self._phase = Phase.MOVING_BLOCK
                    else:
                        self._phase = Phase.SEARCH_ROOM
                else:
                    # Randomly pick a closed door
                    self._door = random.choice(closedDoors)
                    self._phase = Phase.FOLLOW_PATH_TO_CLOSED_DOOR
                doorLoc = self._door['location']
                # Location in front of door is south from door
                doorLoc = doorLoc[0], doorLoc[1] + 1
                # Send message of current action
                self._sendMessage('Moving to door of ' + self._door['room_name'], agent_name)
                self._navigator.add_waypoints([doorLoc])

            if Phase.FOLLOW_PATH_TO_CLOSED_DOOR == self._phase:

                self._state_tracker.update(state)
                # Follow path to door
                action = self._navigator.get_move_action(self._state_tracker)
                if action is not None:
                    return action, {}
                self._phase = Phase.OPEN_DOOR

            if Phase.OPEN_DOOR == self._phase:

                # self._can_be_lazy = True
                self._navigator.reset_full()
                contents = state.get_room_objects(self._door['room_name'])
                waypoints = []
                for c in contents:
                    if "wall" not in c['name']:
                        x, y = c["location"][0], c["location"][1]
                        waypoints.append((x, y))

                self._navigator.add_waypoints(waypoints)

                # Open door
                is_open = state.get_room_doors(self._door['room_name'])[0]['is_open']
                if not is_open:
                    self._sendMessage("Opening door of: " + self._door['room_name'], agent_name)
                    self._sendMessage("Searching through " + self._door['room_name'], agent_name)
                    return OpenDoorAction.__name__, {'object_id': self._door['obj_id']}
                self._phase = Phase.SEARCH_ROOM
            if Phase.SEARCH_ROOM == self._phase:
                if self.__is_lazy() and self._can_be_lazy:
                    self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR
                else:
                    self._can_be_lazy = True

                self._state_tracker.update(state)
                contents = state.get_room_objects(self._door['room_name'])
                for c in contents:
                    # if "Block" in c['name']:
                    #     self._objects.append(c)
                    #     self._sendMessage("Found block of color:" + str(c['visualization']['colour']) + 'and shape: '+ str(c['visualization']['shape']), agent_name)
                    for i in range(len(self._goal_objects)):
                        if c['visualization']['colour'] == self._goal_objects[i]['visualization']['colour'] and \
                                c['visualization']['shape'] == self._goal_objects[i]['visualization']['shape'] and \
                                c['visualization']['size'] == self._goal_objects[i]['visualization']['size'] and \
                                not c['is_goal_block'] and not c['is_drop_zone']:
                            if i == 0:
                                print('found')
                                self._sendMessage(
                                    "Found goal block " + json.dumps(c['visualization']) + " at location " + json.dumps(
                                        c['location']), agent_name)
                                # print(self._door['room_name'])
                                # print(c)
                                self._phase = Phase.MOVE_TO_OBJECT
                                self._current_obj = c
                                self._navigator.reset_full()
                                self._navigator.add_waypoints([c['location']])
                                action = self._navigator.get_move_action(self._state_tracker)
                                return action, {}

                            else:
                                self._goal_objects_found.append(c)

                action = self._navigator.get_move_action(self._state_tracker)

                if action is not None:
                    return action, {}
                self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR

            if Phase.MOVE_TO_OBJECT == self._phase:

                print("moving to obj")
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action is not None:
                    return action, {}
                self._phase = Phase.GRAB
                # self._can_be_lazy = True
                self._sendMessage(
                    "Picking up goal block " + str(self._current_obj['visualization']) + " at location " + str(
                        self._current_obj['location']), agent_name)
                return GrabObject.__name__, {'object_id': self._current_obj['obj_id']}

            if Phase.GRAB == self._phase:
                print("ggrabbing")

                self._navigator.reset_full()
                self._navigator.add_waypoints([self._goal_objects[0]['location']])
                self._phase = Phase.MOVING_BLOCK

            if Phase.MOVING_BLOCK == self._phase:

                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                # print("moving to block")

                if action is not None:
                    # print("is not none moving block")
                    return action, {}

                # if len(self._goal_objects) == 0:
                #     self._phase = Phase.STOP
                if state[agent_name]['is_carrying']:
                    print("dropppgbg")
                    self._goal_objects.remove(self._goal_objects[0])
                    self._can_be_lazy = True
                    self._sendMessage(
                        "Dropping goal block " + str(self._current_obj['visualization']) + " at drop location " +
                        str(self._current_obj['location']), agent_name)
                    return DropObject.__name__, {'object_id': self._current_obj['obj_id']}

                if self._goal_objects and self._goal_objects_found:
                    found = False
                    for o in self._goal_objects_found:
                        if o['visualization']['colour'] == self._goal_objects[0]['visualization']['colour'] and \
                                o['visualization']['shape'] == self._goal_objects[0]['visualization']['shape']:
                            self._navigator.reset_full()
                            self._goal_objects_found.remove(o)
                            self._navigator.add_waypoints([o['location']])
                            self._phase = Phase.MOVE_TO_OBJECT
                            self._current_obj = o
                            print('moving to found one')
                            found = True
                    if not found:
                        self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR
                else:
                    self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR

            if Phase.STOP == self._phase:
                self._phase = Phase.STOP

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

    def __is_lazy(self):
        return random.randint(0, 1) == 1
