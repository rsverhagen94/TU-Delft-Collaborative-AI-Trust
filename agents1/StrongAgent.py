from typing import final, List, Dict, Final
import enum, random
from bw4t.BW4TBrain import BW4TBrain
from matrx.agents.agent_utils.state import State
from matrx.agents.agent_utils.navigator import Navigator
from matrx.agents.agent_utils.state_tracker import StateTracker
from matrx.actions.door_actions import OpenDoorAction
from matrx.actions.object_actions import GrabObject, DropObject
from matrx.messages.message import Message
import pandas as pd
import numpy as np
import re
import ast


class Phase(enum.Enum):
    PLAN_PATH_TO_CLOSED_DOOR = 1,
    FOLLOW_PATH_TO_CLOSED_DOOR = 2,
    OPEN_DOOR = 3,
    ENTER_ROOM = 4,
    TRAVERSE_ROOM = 5,
    DELIVER_ITEM = 6,
    FOLLOW_PATH_TO_DROP_OFF_LOCATION = 7,
    DROP_OBJECT = 8


class StrongAgent(BW4TBrain):

    def __init__(self, settings: Dict[str, object]):
        super().__init__(settings)
        self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR
        self._teamMembers = []
        self.desired_objects = []
        self.agent_name = None
        # only the strong agents can pick 2 blocks
        # for other agents this is 0 or 1
        self.capacity = 0
        self.drop_off_locations = []
        self.object_to_be_dropped = None
        self.initialization_flag = True
        # Stores desired objects
        self.memory = []
        # Stores all seen objects
        self.seenObjects = []
        self.all_rooms = []
        self.ticks = 0
        self.receivedMessages = {}
        self.totalMessagesReceived = 0
        # A list of messages To Be Verified
        self.tbv = []
        # For each team member store trust score
        self.trustBeliefs = {}

    def initialize(self):
        super().initialize()
        self._state_tracker = StateTracker(agent_id=self.agent_id)
        self._navigator = Navigator(agent_id=self.agent_id,
                                    action_set=self.action_set, algorithm=Navigator.A_STAR_ALGORITHM)

    def filter_bw4t_observations(self, state):
        return state

    def decide_on_bw4t_action(self, state: State):
        self.ticks = self.ticks + 1

        # Process messages from team members
        self._processMessages()

        # Update trust beliefs for team members
        self._trustBelief(self._teamMembers, self.receivedMessages)

        # We check if we enter for first time in the method as there is recursion
        # We want to keep track of some objects and reinitialize them every time
        if self.initialization_flag:

            agent_name = state[self.agent_id]['obj_id']
            self.agent_name = agent_name
            # Add team members
            for member in state['World']['team_members']:
                if member != agent_name and member not in self._teamMembers:
                    self._teamMembers.append(member)
                    self.receivedMessages[member] = []

            # Add all rooms in a list
            self.all_rooms = sorted(state.get_all_room_names())

            # Add all desired objects to a list
            desired_objects = list(map(
                lambda x: x, [wall for wall in state.values() if
                              'class_inheritance' in wall and 'GhostBlock' in wall['class_inheritance']]))
            found_obj = []
            # Will not enter here after setting the flag to False
            self.initialization_flag = False

            # Add location for every desired object
            for obj in desired_objects:
                found_obj.append((obj["visualization"], obj["location"]))
            self.desired_objects = sorted(found_obj, key=lambda x: x[1], reverse=True)

            self.initTrustBeliefs()
            self._init_trust_table(state['World']['team_members'])

        while True:

            # Phase entering room
            if Phase.ENTER_ROOM == self._phase:
                # Get the room name for the latest chosen room from the phase PLAN_PATH_TO_CLOSED_DOOR
                room = self._door['room_name']
                self._messageMoveRoom(room)
                # Find all area tiles locations of the room to traverse
                area = list(map(
                    lambda x: x["location"],
                    [wall for wall in state.get_room_objects(room)
                     if 'class_inheritance' in wall and 'AreaTile' in wall['class_inheritance'] and
                     ("is_drop_zone" not in wall or wall['is_drop_zone'] is False)]))

                # Sort the location of the tiles and traverse them
                sorted_by_xy = sorted(sorted(area, key=lambda x: x[1]))
                room = self._door['room_name']
                self._messageSearchThrough(room)
                # Add the locations of the tiles to traverse in order to the navigator
                self._navigator.reset_full()
                self._navigator.add_waypoints(sorted_by_xy)

                # Go to the next phase
                self._phase = Phase.TRAVERSE_ROOM

            if Phase.TRAVERSE_ROOM == self._phase:
                # Every time update the state for the new location of the agent
                self._state_tracker.update(state)

                action = self._navigator.get_move_action(self._state_tracker)
                # If the agent has moved update look for and item
                # We are interested only in collectable items (such that can be picked)
                object_prop = list(map(
                    lambda x: x, [wall for wall in state.get_closest_with_property("is_collectable") if
                                  wall["is_collectable"] is True and not 'GhostBlock' in wall[
                                      'class_inheritance']]))

                # For all possible objects save only visualization and id
                found_obj = []
                for obj in object_prop:
                    found_obj.append((obj["visualization"], obj["obj_id"], obj["location"]))

                # Check if some of the found objects that can be collected are desired objects
                for obj in found_obj:
                    self.addToSeenObjects((obj[0], obj[2]))
                    self._messageFoundBlock(str(obj[0]), str(obj[2]))
                    for des, loc in self.desired_objects:
                        if obj[0]["shape"] == des["shape"] and obj[0]["colour"] == des["colour"]:
                            # In case they are desired objects for the strong agent we are interested only in the
                            # first two items from bottom to up, if they are we pick them
                            # in case they are not we save them in the memory for later use
                            self._messageFoundGoalBlock(str(obj[0]), str(obj[2]))

                            if (des, loc) in self.desired_objects[0:(2 - self.capacity)]:
                                if self.capacity < 2:
                                    self.capacity += 1
                                    self.drop_off_locations.append((obj[0], obj[1], loc))
                                    self.desired_objects.remove((des, loc))
                                    self._messagePickUpGoalBlock(str(obj[0]), str(obj[2]))

                                    for num, dict1 in enumerate(self.memory):
                                        if obj[0]["shape"] == dict1["visualization"]["shape"] \
                                                and obj[0]["colour"] == dict1["visualization"]["colour"]:
                                            self.memory.remove(dict1)

                                    return GrabObject.__name__, {'object_id': obj[1]}
                                else:
                                    self.addToMemory(obj[0], obj[2], loc)

                            elif (des, loc) in self.desired_objects:
                                # Note a small bug was found. It does not find and pick object
                                # when the memory is pointing to the middle room (room 5).
                                # In all other cases it work properly
                                # Grab object if there is a capacity

                                self.addToMemory(obj[0], obj[2], loc)

                # In case we are filled, deliver items, next phase
                if self.capacity > 1:
                    # print("Deliver 2")
                    self._phase = Phase.DELIVER_ITEM
                # In case there is only one object left needed and is found deliver it, next phase
                elif len(self.desired_objects) < 2 and self.capacity > 0:
                    # print("Get block")
                    self._phase = Phase.DELIVER_ITEM

                    # If no desired object was found just move
                if action is not None:
                    return action, {}
                elif self._phase != Phase.DELIVER_ITEM:
                    # If the room is traversed go to te next room
                    self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR

            # Find the path to the deliver location
            if Phase.DELIVER_ITEM == self._phase:
                locations = []
                # sort the location of the picked items so that the first dropped will be at the bottom
                for _, _, loc in self.drop_off_locations:
                    locations.append(loc)
                locations.sort(reverse=True)
                self._navigator.reset_full()
                # Add the navigation
                # print(locations)
                self._navigator.add_waypoints(locations)

                # Next phase
                self._phase = Phase.FOLLOW_PATH_TO_DROP_OFF_LOCATION

            # Follow path to the drop off location
            if Phase.FOLLOW_PATH_TO_DROP_OFF_LOCATION == self._phase:
                flag = False
                # Check if the current location of the agent is the correct drop off location
                # print("WAYPOINTS", self._navigator.get_all_waypoints())
                for obj_viz, obj_id, loc in self.drop_off_locations:
                    if state[self._state_tracker.agent_id]['location'] == loc:
                        flag = True
                        self.object_to_be_dropped = obj_id
                        # if it is the correct location drop the object
                        self._phase = Phase.DROP_OBJECT
                        self.drop_off_locations.remove((obj_viz, obj_id, loc))
                        self._messageDroppedGoalBlock(str(obj_viz), str(loc))

                # if not already dropped the object move to the next location
                if not flag:
                    self._state_tracker.update(state)

                    action = self._navigator.get_move_action(self._state_tracker)
                    # Move to the next location
                    if action != None:
                        return action, {}
                    else:
                        self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR

                # print("! DONE !")

            if Phase.DROP_OBJECT == self._phase:
                if self.object_to_be_dropped is None:
                    print("CODE BROKEN VERY BAD")
                    exit(-1)
                # update capacity
                self.capacity -= 1
                # print("dropped object")
                # Drop object
                self._phase = Phase.FOLLOW_PATH_TO_DROP_OFF_LOCATION

                return DropObject.__name__, {'object_id': self.object_to_be_dropped}

            if Phase.PLAN_PATH_TO_CLOSED_DOOR == self._phase:
                self._navigator.reset_full()

                closedDoors = [door for door in state.values()
                               if 'class_inheritance' in door and 'Door' in door['class_inheritance'] and not door[
                        'is_open']]
                if len(self.memory) > 0 and ((self.desired_objects[0][0]["shape"] ==
                                              self.memory[0]["visualization"]["shape"]
                                              and self.desired_objects[0][0]["colour"] ==
                                              self.memory[0]["visualization"]["colour"]) or
                                             (len(self.memory) > 1 and
                                              self.desired_objects[0][1]["shape"] ==
                                              self.memory[0]["visualization"]["shape"]
                                              and self.memory[0]["visualization"]["colour"] ==
                                              self.desired_objects[0][1]["colour"])):

                    # print("MEMORY", self.memory)
                    self._navigator.reset_full()
                    self._navigator.add_waypoints([self.memory[0]["location"]])

                    self._phase = Phase.TRAVERSE_ROOM
                # Randomly pick a closed door or go to open room
                # Check if all rooms open
                else:
                    if len(closedDoors) == 0:
                        # If no rooms - stuck
                        if len(self.all_rooms) == 0:
                            return None, {}
                        # get the first room, as they were sorted in the first iteration
                        room_name = self.all_rooms.pop(0)
                        # get the door of the chosen room
                        self._door = [loc for loc in state.values()
                                      if "room_name" in loc and loc['room_name'] is
                                      room_name and 'class_inheritance' in loc and
                                      'Door' in loc['class_inheritance']]

                        # in case some broken room without door - stuck
                        if len(self._door) == 0:
                            return None, {}
                        else:
                            self._door = self._door[0]

                    # randomly pick closed door
                    else:
                        # self._door = random.choice(closedDoors)
                        self._door = closedDoors[0]

                    # get the location of the door
                    doorLoc = self._door['location']

                    # Location in front of door is south from door
                    doorLoc = doorLoc[0], doorLoc[1] + 1

                    # Send message of current action
                    self._sendMessage('Moving to door of ' + self._door['room_name'], self.agent_name)
                    # self._messageMoveRoom()
                    self._navigator.add_waypoints([doorLoc])
                    # go to the next phase
                    self._phase = Phase.FOLLOW_PATH_TO_CLOSED_DOOR

            if Phase.FOLLOW_PATH_TO_CLOSED_DOOR == self._phase:
                self._state_tracker.update(state)
                # Follow path to door
                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    return action, {}
                # go to the next phase
                self._phase = Phase.OPEN_DOOR

            if Phase.OPEN_DOOR == self._phase:
                self._phase = Phase.ENTER_ROOM
                # Open door
                # If already opened, no change
                self._messageOpenDoor(self._door['room_name'])
                return OpenDoorAction.__name__, {'object_id': self._door['obj_id']}

    def _sendMessage(self, mssg, sender):
        '''
        Enable sending messages in one line of code
        '''
        msg = Message(content=mssg, from_id=sender)
        if msg.content not in self.received_messages:
            self.send_message(msg)

    def _messageMoveRoom(self, room):
        self._sendMessage("Moving to " + room, self.agent_name)

    def _messageOpenDoor(self, room):
        self._sendMessage("Opening door of " + room, self.agent_name)

    def _messageSearchThrough(self, room):
        self._sendMessage("Searching through " + room, self.agent_name)

    def _messageFoundGoalBlock(self, block_visualization, location):
        self._sendMessage("Found goal block " + block_visualization + " at location " + location, self.agent_name)

    def _messagePickUpGoalBlock(self, block_visualization, location):
        self._sendMessage("Picking up goal block " + block_visualization + " at location " + location, self.agent_name)

    def _messageDroppedGoalBlock(self, block_visualization, location):
        self._sendMessage("Dropped goal block " + block_visualization + " at drop location " + location, self.agent_name)

    def _messageFoundBlock(self, block_visualization, location):
        self._sendMessage("Found block " + block_visualization + " at location " + location, self.agent_name)

    def _init_trust_table(self, ids):
        data = {}
        for id in ids:
            arr = np.zeros(len(ids))
            arr.fill(0.5)
            data.update({id: arr})
        df = pd.DataFrame(data, index=ids, dtype=float)
        print(df)
        df.to_csv("Trust.csv")

    def _write_to_trust_table(self, truster_index, trustee, new_trust):
        df = pd.read_csv('Trust.csv')
        df.loc[truster_index, trustee] = new_trust
        print(df)
        df.to_csv('Trust.csv')

    def increaseTrust(self, trustee):
        self.trustBeliefs[trustee] = np.clip(self.trustBeliefs[trustee] + 0.1, 0, 1)

    def decreaseTrust(self, trustee):
        self.trustBeliefs[trustee] = np.clip(self.trustBeliefs[trustee] - 0.1, 0, 1)

    def _processMessages(self):
        '''
        Process incoming messages and create a dictionary with received messages from each team member.
        '''

        for mssg in self.received_messages[self.totalMessagesReceived:]:
            for member in self._teamMembers:
                if mssg.from_id == member:
                    self.receivedMessages[member].append((self.ticks, mssg.content, mssg.from_id))
                    self.totalMessagesReceived = self.totalMessagesReceived + 1
                    self.tbv.append((self.ticks, mssg.content, mssg.from_id))
                    is_true = self.checkMessageTrue(self.ticks, mssg.content, mssg.from_id)
                    print('mssg', mssg.content, '\n', is_true)
        # print('tbv', self.tbv)
        # tbv_copy = self.tbv
        # for (ticks, mssg, from_id) in tbv_copy:
        #     is_true = self.checkMessageTrue(self.ticks, mssg, from_id)
        #     # print(mssg, is_true)
        #     # print(self.seenObjects)
        #     if is_true is not None:
        #         if is_true:
        #             self.increaseTrust(from_id)
        #             print('truth', self.trustBeliefs[from_id])
        #         else:
        #             self.decreaseTrust(from_id)
        #             print('lie', self.trustBeliefs[from_id])
        #         self.tbv.remove((ticks, mssg, from_id))
        #     # print('no info', self.trustBeliefs[from_id])

    def initTrustBeliefs(self):
        for member in self._teamMembers:
            self.trustBeliefs[member] = 0.5

    def _trustBelief(self, member, received):
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

    def addToMemory(self, vis, loc, drop):
        if len(self.memory) == 0:
            self.memory.append({"visualization": vis,
                                "location": loc,
                                "drop_off_location": drop})
        flag_check = True
        for v in self.memory:
            if v["visualization"]["colour"] == vis["colour"] and v["visualization"]["shape"] == vis["shape"]:
                flag_check = False

        if flag_check:
            self.memory.append({"visualization": vis,
                                "location": loc,
                                "drop_off_location": drop})

        self.memory = sorted(self.memory, key=lambda x: x["drop_off_location"],
                             reverse=True)
        # print("MEMORY", self.memory)

    def checkMessageTrue(self, ticks, mssg, sender):
        splitMssg = mssg.split(' ')
        if splitMssg[0] == 'Moving' and splitMssg[1] == 'to':
            pass

        if splitMssg[0] == 'Opening' and splitMssg[1] == 'door':
            room = splitMssg[2]

        if splitMssg[0] == 'Searching' and splitMssg[1] == 'through':
            room = splitMssg[2]

        if splitMssg[0] == 'Found' and splitMssg[1] == 'goal':
            vis, loc = self.getVisLocFromMessage(mssg)
            for obj in self.seenObjects:
                if self.compareObjects(vis, obj[0]) and obj[1] == loc:
                    return True
                elif str(obj[1]) == loc:
                    return False

        if splitMssg[0] == 'Dropped' and splitMssg[1] == 'goal':
            vis, loc = self.getVisLocFromMessage(mssg)

        if splitMssg[0] == 'Picking' and splitMssg[2] == 'goal':
            vis, loc = self.getVisLocFromMessage(mssg)

        if splitMssg[0] == 'Found' and splitMssg[1] == 'block':
            vis, loc = self.getVisLocFromMessage(mssg)
            for obj in self.seenObjects:
                if self.compareObjects(vis, obj[0]) and obj[1] == loc:
                    return True
                elif str(obj[1]) == loc:
                    return False
        return None

    def getVisLocFromMessage(self, mssg):
        bv = re.search("\{(.*)\}", mssg)
        l = re.search("\((.*)\)", mssg)
        vis = None
        loc = None
        if bv is not None:
            vis = ast.literal_eval(mssg[bv.start(): bv.end()])
        if l is not None:
            loc = ast.literal_eval(mssg[l.start(): l.end()])
        return vis, loc

    def compareObjects(self, obj1, obj2):
        for key in obj1:
            if obj1[key] != obj2[key]:
                return False
        return True

    def addToSeenObjects(self, obj):
        if obj not in self.seenObjects:
            self.seenObjects.append(obj)