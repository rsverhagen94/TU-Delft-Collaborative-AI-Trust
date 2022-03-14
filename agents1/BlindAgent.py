import enum
import random
import json
from typing import Dict

from matrx.actions.door_actions import OpenDoorAction
from matrx.agents.agent_utils.navigator import Navigator
from matrx.agents.agent_utils.state import State
from matrx.agents.agent_utils.state_tracker import StateTracker
from matrx.goals import WorldGoalV2
from matrx.messages.message import Message
from matrx.actions.object_actions import GrabObject, DropObject

from matrx.grid_world import GridWorld

from bw4t.BW4TBrain import BW4TBrain


class Phase(enum.Enum):
    PLAN_PATH_TO_CLOSED_DOOR = 1,
    FOLLOW_PATH_TO_CLOSED_DOOR = 2,
    OPEN_DOOR = 3,
    SEARCH_ROOM = 4,

    PICKUP_BLOCK = 5,
    MOVING_BLOCK = 6,
    DROP_BLOCK = 7,

    FOLLOW_PATH_TO_GOAL_BLOCK = 8,



class BlindAgent(BW4TBrain):

    def __init__(self, settings: Dict[str, object]):
        super().__init__(settings)
        self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR
        self._teamMembers = []

    def initialize(self):
        super().initialize()
        self._door = None
        self._state_tracker = StateTracker(agent_id=self.agent_id)
        self._navigator = Navigator(agent_id=self.agent_id,
                                    action_set=self.action_set, algorithm=Navigator.A_STAR_ALGORITHM)
        self._objects = []
        self._goal_blocks = []
        self._goal_blocks_locations = []
        self._goal_blocks_locations_followed = []
        self._trustBeliefs = []

    def filter_bw4t_observations(self, state):
        # TODO: Filter out property "color" from each "visualization"
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
        self._trustBeliefs = self._trustBlief(self._teamMembers, receivedMessages)
        self.updateGoalBlocks(state)

        # Remember locations of goal blocks (as said by others in messages)
        self._findGoalBlocksInMessages(receivedMessages, self._teamMembers)

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
                print('Moving to door of ' + str(self._door['room_name']))
                self._navigator.add_waypoints([doorLoc])
                self._phase = Phase.FOLLOW_PATH_TO_CLOSED_DOOR

            if Phase.FOLLOW_PATH_TO_CLOSED_DOOR == self._phase:
                print("Follow path to closed door")
                self._state_tracker.update(state)
                # Follow path to door
                action = self._navigator.get_move_action(self._state_tracker)
                if action is not None:
                    return action, {}
                self._phase = Phase.OPEN_DOOR

            if Phase.OPEN_DOOR == self._phase:
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
                    self._phase = Phase.SEARCH_ROOM
                    self._sendMessage("Searching through " + self._door['room_name'], agent_name)
                    print("Searching through " + str(self._door['room_name']))
                    return OpenDoorAction.__name__, {'object_id': self._door['obj_id']}

            if Phase.SEARCH_ROOM == self._phase:
                self._state_tracker.update(state)

                contents = state.get_room_objects(self._door['room_name'])

                for c in contents:
                    if "Block" in c['name']:
                        if c not in self._objects:
                            self._objects.append(c)

                            message = "Found block {\"size\": " + str(c['visualization']['size']) + ", \"shape\": " + \
                                      str(c['visualization']['shape']) + "} at location " + str(c['location'])
                            self._sendMessage(message, agent_name)

                action = self._navigator.get_move_action(self._state_tracker)
                if action is not None:
                    return action, {}

                # After searching room, either follow path to goal block or plan path to closed door
                follow = None
                for loc in self._goal_blocks_locations:
                    if loc not in self._goal_blocks_locations_followed:
                        follow = loc['location']
                        self._goal_blocks_locations_followed.append(loc)
                        break

                # There is a goal block
                if follow is not None:
                    self._navigator.add_waypoint(follow)
                    self._phase = Phase.FOLLOW_PATH_TO_GOAL_BLOCK

                # There is no goal block
                else:
                    self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR

            if Phase.FOLLOW_PATH_TO_GOAL_BLOCK == self._phase:
                self._state_tracker.update(state)

                action = self._navigator.get_move_action(self._state_tracker)

                if action is not None:
                    return action, {}

                # Here pick up goal block and move it (close) to drop location
                # For now, plan path to closed door
                self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR

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

    def _findGoalBlocksInMessages(self, receivedMessages, teamMembers):
        # If another agent found goal block, check other attributes that we know and go pick it up
        for member in teamMembers:
            messages = receivedMessages[member]
            for mess in messages:
                if mess[0:16] == "Found goal block":

                    charac = json.loads(mess[18:mess.find("at")])
                    loc_str = mess[mess.find("location") + 9:len(mess)]
                    location = (int(loc_str[1:loc_str.find(",")]), int(loc_str[loc_str.find(",") + 2:len(loc_str)-1]))

                    # Check that agent didn't lie about size, shape and color of goal block
                    for goal in self._goal_blocks:
                        if goal['visualization']['size'] == charac['size'] and goal['visualization']['shape'] == charac['shape'] and \
                                goal['visualization']['colour'] == charac['colour']:

                            # Save goal block locations as mentioned by other agents
                            # Location + member that sent message + trust in member
                            obj = {
                                "location": location,
                                "member": member,
                                "trustLevel": self._trustBeliefs[member]
                            }
                            if obj not in self._goal_blocks_locations:
                                self._goal_blocks_locations.append(obj)
                                self._goal_blocks_locations.sort(key=lambda x: x['trustLevel'], reverse=True)



    def updateGoalBlocks(self, state):
        if len(self._goal_blocks) == 0:
            self._goal_blocks = [goal for goal in state.values()
                        if 'is_goal_block' in goal and goal['is_goal_block']]

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
