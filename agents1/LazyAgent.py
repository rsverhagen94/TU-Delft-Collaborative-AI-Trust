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
    GO_TO_REORDER_ITEMS = 9
    REORDER_ITEMS = 10
    GRAB_AND_DROP = 11
    CHECK_ITEMS = 12


class LazyAgent(BW4TBrain):

    def __init__(self, settings: Dict[str, object]):
        super().__init__(settings)
        self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR
        self._teamMembers = []
        self.desired_objects = []
        self.all_desired_objects = []

        self.agent_name = None
        # for other agents this is 0 or 1
        self.capacity = 0
        self.drop_off_locations = None
        self.object_to_be_dropped = None
        self.initialization_flag = True
        self.memory = []
        self.all_rooms = []
        self.decision = -1
        self.stop_when = 0
        self.my_object = None
        self.use_memory = True
        self.seenObjects = []
        self.ticks = 0
        self.receivedMessages = {}
        self.totalMessagesReceived = 0
        self.tbv = []
        self.trustBeliefs = {}
        self.rooms_to_visit = []
        self.visited = []
        self._door = None
        self.dropped_off_count = 0
        self.at_drop_location = {}
        self.checked_room = None
        # used for the last phase - GRAB_AND_DROP to keep track of when an object is grabbed and after it was just dropped

        self.grab = False
        self.drop = False
        self.closed_doors = []
        self.obj_id = None
        self.delivered_item = False

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

        self.believeAgent()

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

            # Process messages from team members
            #receivedMessages = self._processMessages(self._teamMembers)
            # Update trust beliefs for team members
            # Add all rooms in a list
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
                self.at_drop_location[obj["location"]] = 0

            self.desired_objects = sorted(found_obj, key=lambda x: x[1], reverse=True)
            self.initTrustBeliefs()
            self._init_trust_table(state['World']['team_members'])
            self.all_desired_objects = self.desired_objects.copy()
            sorted(self.all_desired_objects, key=lambda obj: obj[1], reverse=True)


            for room in state.get_all_room_names():
                if room != "world_bounds":
                    # print(room)
                    door = state.get_room_doors(room)
                    self.rooms_to_visit.append((room, door[0]))
                    self.all_rooms.append(room)

            self.closed_doors = [door for door in state.values()
                            if 'class_inheritance' in door and 'Door' in door['class_inheritance'] and not door[
                    'is_open']]

        while True:

            # Phase entering room
            if Phase.ENTER_ROOM == self._phase:
                # Get the room name for the latest chosen room from the phase PLAN_PATH_TO_CLOSED_DOOR
                room = self._door['room_name']
                # Find all area tiles locations of the room to traverse
                area = list(map(
                    lambda x: x["location"],
                    [wall for wall in state.get_room_objects(room)
                     if 'class_inheritance' in wall and 'AreaTile' in wall['class_inheritance'] and
                     ("is_drop_zone" not in wall or wall['is_drop_zone'] is False)]))

                # Sort the location of the tiles and traverse them
                sorted_by_xy = sorted(sorted(area, key=lambda x: x[1]))

                # Add the locations of the tiles to traverse in order to the navigator
                self._navigator.reset_full()
                decision = self.getRandom50()
                print("TRAVERSE ROOM: ", decision)

                room = self._door['room_name']
                self._messageSearchThrough(room)
                if decision:
                    self._navigator.add_waypoints(sorted_by_xy)
                    self._phase = Phase.TRAVERSE_ROOM
                    self.stop_when = -1
                    self.rooms_to_visit.remove(self.checked_room)
                else:
                    self.stop_when = int(round(len(sorted_by_xy) * self.getRandom1()))
                    self._navigator.add_waypoints(sorted_by_xy)
                    self._phase = Phase.TRAVERSE_ROOM
                # Go to the next phase

            if Phase.TRAVERSE_ROOM == self._phase:
                # Every time update the state for the new location of the agent
                # room = self._door['room_name']
                # self._messageSearchThrough(room)

                self._state_tracker.update(state)

                action = self._navigator.get_move_action(self._state_tracker)

                if self.stop_when > 0 or self.stop_when == -1:

                    if self.stop_when > 0:
                        self.stop_when -= 1
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
                    check = False

                    for obj in found_obj:
                        self.addToSeenObjects((obj[0], obj[2]))

                        for des, loc in self.desired_objects:
                            if obj[0]["shape"] == des["shape"] and obj[0]["colour"] == des["colour"]:
                                # In case they are desired objects for the strong agent we are interested only in the
                                # first two items from bottom to up, if they are we pick them
                                # in case they are not we save them in the memory for later use
                                self._messageFoundGoalBlock(str(obj[0]), str(loc))
                                # print(self.desired_objects[0])
                                # print((des, loc))

                                if ((des, loc)) in self.desired_objects:
                                    decision = self.getRandom50()
                                    print("PICK AN ITEM, DECISION", decision)

                                    # Grab object if there is a capacity
                                    if self.capacity < 1:
                                        if decision:
                                            self.capacity += 1
                                            self.drop_off_locations = (obj[0], obj[1], loc)
                                            self.my_object = ((des, loc))
                                            self.my_object_id = obj[1]
                                            self._messagePickUpGoalBlock(str(obj[0]), str(loc))
                                            self._phase = Phase.DELIVER_ITEM

                                            for dict1 in self.memory:
                                                if obj[0]["shape"] == dict1["visualization"]["shape"] \
                                                        and obj[0]["colour"] == dict1["visualization"]["colour"]:
                                                    self.memory.remove(dict1)

                                            return GrabObject.__name__, {'object_id': obj[1]}

                                        self.addToMemory(obj[0], obj[2], loc)

                                elif ((des, loc)) in self.desired_objects:
                                    self.addToMemory(obj[0], obj[2], loc)

                                    print("MEMORY", self.memory)

                        if not check:
                            self._messageFoundBlock(str(obj[0]), str(obj[2]))
                            check = False

                    # If no desired object was found just move
                    if action != None:
                        return action, {}

                # If the room is traversed go to te next room
                self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR

            # Find the path to the deliver location
            if Phase.DELIVER_ITEM == self._phase:
                # sort the location of the picked items so that the first dropped will be at the bottom

                self._navigator.reset_full()
                # Add the navigation

                decision = self.getRandom50()
                self._navigator.add_waypoints([self.drop_off_locations[2]])
                print("DELIVER ITEM: ", decision)
                if decision:
                    self.stop_when = -1
                else:
                    current_loc = state[self._state_tracker.agent_id]['location']
                    distance = self.shortestDistance(current_loc, self.drop_off_locations[2])
                    self.stop_when = int(round(distance * self.getRandom1()))

                # Next phase
                self._phase = Phase.FOLLOW_PATH_TO_DROP_OFF_LOCATION

            # Follow path to the drop off location
            if Phase.FOLLOW_PATH_TO_DROP_OFF_LOCATION == self._phase:
                flag = False
                self.delivered_item = False

                # Check if the current location of the agent is the correct drop off location
                # print(self.drop_off_locations)

                if self.stop_when == 0:
                    self._phase = Phase.DROP_OBJECT
                    self._messageDroppedGoalBlock(str(self.my_object[0]),
                                                  str(state[self._state_tracker.agent_id]['location']))
                    self.addToMemory(self.my_object[0], state[self._state_tracker.agent_id]['location'],
                                     self.my_object[1])

                    self.object_to_be_dropped = self.my_object_id

                elif state[self._state_tracker.agent_id]['location'] == self.drop_off_locations[2]:
                    flag = True

                    self.object_to_be_dropped = self.drop_off_locations[1]
                    self.delivered_item = True

                    # if it is the correct location drop the object

                    self._phase = Phase.DROP_OBJECT

                    self.desired_objects.remove((self.my_object[0], self.my_object[1]))

                    self._messageDroppedGoalBlock(str(self.drop_off_locations[0]), str(self.drop_off_locations[2]))

                    #print("MEMORY", self.memory)

                # if not already dropped the object move to the next location
                if not flag and (self.stop_when == -1 or self.stop_when > 0):
                    if self.stop_when > 0:
                        self.stop_when -= 1
                    self._state_tracker.update(state)

                    action = self._navigator.get_move_action(self._state_tracker)
                    # Move to the next location
                    if action != None:
                        return action, {}
                    else:
                        # If dropped both items use the memory to go to the next desired object, that was found
                        # Use the traverse method phase for now and check on every step
                        # could be implemented to go to the room and then traverse it again
                        # now just checks every step
                        # If memory is empty continue traversing rooms
                        self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR

            if Phase.DROP_OBJECT == self._phase:
                if self.object_to_be_dropped is None:
                    print("CODE BROKEN VERY BAD")
                    exit(-1)
                # update capacity
                self.capacity -= 1
                # Drop object

                # print("DELIVER ITEMSSS: ", self.delivered_item)
                if self.delivered_item:
                    locations = list(map(lambda des_obj: des_obj[1], self.all_desired_objects))
                    self._navigator.reset_full()
                    # Add the navigation
                    self._navigator.add_waypoints(locations)
                    self.dropped_off_count = 0
                    self._phase = Phase.CHECK_ITEMS
                else:
                    self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR
                    self.delivered_item = False



                # if len(self.desired_objects) == 0:
                #     self._phase = Phase.GO_TO_REORDER_ITEMS

                return DropObject.__name__, {'object_id': self.object_to_be_dropped}

            if Phase.PLAN_PATH_TO_CLOSED_DOOR == self._phase:
                self._navigator.reset_full()
                decision = self.getRandom50()
                print("USE MEMORY", self.use_memory)
                doc = 0
                for key in self.at_drop_location:
                    doc += self.at_drop_location[key]
                if doc == len(self.all_desired_objects):
                    # print(self.dropped_off_count)
                    self._phase = Phase.GO_TO_REORDER_ITEMS
                    return None, {}


                if len(self.memory) > 0 and self.use_memory:

                    print("GO TO MEMORY: ", decision)
                    self._navigator.reset_full()
                    self._navigator.add_waypoints([self.memory[0]["location"]])

                    if decision:
                        self.stop_when = -1
                        self.use_memory = True
                        print("GO TO OBJECT", self.memory[0])
                        self.memory.pop(0)
                    else:
                        distance = self.getRandom1() * self.shortestDistance(
                            state[self._state_tracker.agent_id]['location'], self.memory[0]["location"])

                        self.stop_when = int(round(distance * self.getRandom1()))
                        print("STOP when", self.stop_when)
                        self.use_memory = False
                    self._phase = Phase.TRAVERSE_ROOM
                else:
                    self.use_memory = True

                    if len(self.rooms_to_visit) != 0:
                        self.checked_room = self.rooms_to_visit.pop(random.randint(0, len(self.rooms_to_visit) - 1))
                        self._door = self.checked_room[1]
                        self.rooms_to_visit.append(self.checked_room)
                    elif len(self.visited) != 0:
                        self._door = self.visited.pop()[1]
                    else:
                        if len(self.all_rooms) == 0:
                            return None, {}
                        # get the first room, as they were sorted in the first iteration
                        room_name = self.all_rooms.pop(0)
                        self.all_rooms.append(room_name)
                        # get the door of the chosen room
                        # print("ROOM", room_name)
                        self._door = [loc for loc in state.values()
                                      if "room_name" in loc and loc['room_name'] is
                                      room_name and 'class_inheritance' in loc and
                                      'Door' in loc['class_inheritance']][0]

                        # in case some broken room without door - stuck
                        if len(self._door) == 0:
                            return None, {}
                        # else:
                        #     self._door = self._door

                    doorLoc = self._door["location"]
                    # Location in front of door is south from door
                    doorLoc = (doorLoc[0], doorLoc[1] + 1)

                    # Send message of current action
                    # self._sendMessage('Moving to door of ' + self._door['room_name'], self.agent_name)
                    self._messageMoveRoom(self._door['room_name'])
                    self._navigator.add_waypoints([doorLoc])
                    # go to the next phase
                    # self._messageMoveRoom(room)
                    self._phase = Phase.FOLLOW_PATH_TO_CLOSED_DOOR

                    decision = self.getRandom50()
                    print("GO TO ROOM: ", decision)

                    if decision:
                        self._messageMoveRoom(self._door['room_name'])
                        #self._sendMessage('Moving to door of ' + self._door['room_name'], agent_name)
                        # go to the next phase
                        self._phase = Phase.FOLLOW_PATH_TO_CLOSED_DOOR
                        self.stop_when = -1
                    else:
                        current_loc = state[self._state_tracker.agent_id]['location']
                        distance = self.shortestDistance(current_loc, doorLoc)
                        self.stop_when = int(round(distance * self.getRandom1()))
                        self._phase = Phase.FOLLOW_PATH_TO_CLOSED_DOOR

            # When going using memory randomize

            if Phase.FOLLOW_PATH_TO_CLOSED_DOOR == self._phase:
                if self.stop_when > 0:
                    self.stop_when -= 1
                    self._state_tracker.update(state)
                    # Follow path to door
                    action = self._navigator.get_move_action(self._state_tracker)
                    if action != None:
                        return action, {}
                    # go to the next phase
                    self._phase = Phase.OPEN_DOOR
                elif self.stop_when == -1:
                    self._state_tracker.update(state)
                    # Follow path to door
                    action = self._navigator.get_move_action(self._state_tracker)
                    if action != None:
                        return action, {}
                    # go to the next phase
                    self._phase = Phase.OPEN_DOOR
                else:
                    self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR

            if Phase.OPEN_DOOR == self._phase:
                self._phase = Phase.ENTER_ROOM
                # Open door
                # If already opened, no change
                self._messageOpenDoor(self._door['room_name'])

                decision = self.getRandom50()
                print("OPEN DOOR: ", decision)

                if decision:
                    return OpenDoorAction.__name__, {'object_id': self._door['obj_id']}
                else:
                    self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR

            if Phase.GO_TO_REORDER_ITEMS == self._phase:
                locations = []
                # sort the location of the picked items so that the first dropped will be at the bottom
                locations = list(map(lambda des_obj: des_obj[1], self.all_desired_objects))
                self._navigator.reset_full()
                # Add the navigation
                self._navigator.add_waypoints(locations)

                self._phase = Phase.REORDER_ITEMS

            if Phase.REORDER_ITEMS == self._phase:
                if state[self._state_tracker.agent_id]['location'] == self.all_desired_objects[0][1]:
                    self.all_desired_objects.pop(0)
                    self._phase = Phase.GRAB_AND_DROP

                if self._phase != Phase.GRAB_AND_DROP:
                    self._state_tracker.update(state)
                    action = self._navigator.get_move_action(self._state_tracker)
                    # Move to the next location
                    if action != None:
                        return action, {}
                    else:
                        print("SHOULD BE DONE!")

            if Phase.GRAB_AND_DROP == self._phase:
                if not self.grab:
                    self.obj_id = self.getObjectIdFromLocation(state, state[self._state_tracker.agent_id]['location'])
                    self.grab = True
                    return GrabObject.__name__, {'object_id': self.obj_id}
                if not self.drop:
                    self.drop = True
                    return DropObject.__name__, {'object_id': self.obj_id}

                self.grab = False
                self.drop = False
                self._phase = Phase.REORDER_ITEMS

            if Phase.CHECK_ITEMS == self._phase:
                #print("IN CHECK ITEMS")
                # Every time update the state for the new location of the agent
                self._state_tracker.update(state)

                action = self._navigator.get_move_action(self._state_tracker)

                # print("tuk printq", self.dropped_off_count, self.agent_name)
                # print(state[self._state_tracker.agent_id]['location'], self.capacity)
                for des, loc in self.all_desired_objects:
                    myLoc = state[self._state_tracker.agent_id]['location']
                    if myLoc == loc:
                        for obj in state.get_closest_with_property("is_collectable"):
                            if obj["is_collectable"] is True and not 'GhostBlock' in obj['class_inheritance'] and obj[
                                "location"] == myLoc:
                                if self.compareObjects(des, obj):
                                    self.at_drop_location[loc] = 1
                                    break
                        else:
                            continue
                        break
                if not found:
                    for des, loc in self.all_desired_objects:
                        if myLoc == loc:
                            if (des, loc) not in self.desired_objects:
                                print('Increase desired obj')
                                self.desired_objects.append((des, loc))
                if action is not None:
                    return action, {}
                else:
                    self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR

    def getRandom50(self):
        return random.random() > 0.5

    def getRandom1(self):
        return random.random()

    def shortestDistance(self, a, b):
        if a[0] > b[0]:
            x = a[0] - b[0]
        else:
            x = b[0] - a[0]

        if a[1] > b[1]:
            y = a[1] - b[1]
        else:
            y = b[1] - a[1]

        return (x ** 2 + y ** 2) ** 0.5

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

    def getObjectIdFromLocation(self, state, loc):
        for obj in state.get_closest_with_property("is_collectable"):
            if obj["is_collectable"] is True and not 'GhostBlock' in obj['class_inheritance'] and obj["location"] == loc:
                return obj["obj_id"]

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

    def _write_to_trust_table(self, trustor, trustee, new_trust):
        df = pd.read_csv('Trust.csv', index_col=0)
        df.loc[trustor, trustee] = new_trust
        print(df)
        df.to_csv('Trust.csv')

    def increaseTrust(self, trustee):
        self.trustBeliefs[trustee] = np.clip(self.trustBeliefs[trustee] + 0.1, 0, 1)
        self._write_to_trust_table(self.agent_id, trustee, self.trustBeliefs[trustee])

    def decreaseTrust(self, trustee):
        self.trustBeliefs[trustee] = np.clip(self.trustBeliefs[trustee] - 0.1, 0, 1)
        self._write_to_trust_table(self.agent_id, trustee, self.trustBeliefs[trustee])

    def _processMessages(self):
        '''
        Process incoming messages and create a dictionary with received messages from each team member.
        '''

        for mssg in self.received_messages[self.totalMessagesReceived:]:
            for member in self._teamMembers:
                if mssg.from_id == member:
                    self.receivedMessages[member].append((self.ticks, mssg.content, False))
                    self.totalMessagesReceived = self.totalMessagesReceived + 1
                    if (self.ticks, mssg.content, mssg.from_id) not in self.tbv:
                        self.tbv.append((self.ticks, mssg.content, mssg.from_id))
                    self.acceptMessageIfSenderTrustworthy(mssg.content, mssg.from_id)
                    is_sequence_true = self.verify_action_sequence(self.receivedMessages, member, self.closed_doors)
                    if is_sequence_true is not None:
                        if is_sequence_true:
                            self.increaseTrust(member)
                        else:
                            self.decreaseTrust(member)
        tbv_copy = self.tbv
        for (ticks, mssg, from_id) in tbv_copy:
            is_true = self.checkMessageTrue(self.ticks, mssg, from_id)
            if is_true is not None:
                if is_true:
                    self.increaseTrust(from_id)
                else:
                    self.decreaseTrust(from_id)
                self.tbv.remove((ticks, mssg, from_id))

    def believeAgent(self):
        for agent in self.receivedMessages:
            if self.trustBeliefs[agent] >= 0.9:
                pass
                # for i in range(len(self.receivedMessages[agent])):
                #     mssg = self.receivedMessages[agent][i]
                #     if not mssg[2]:
                #         self.acceptMessageIfSenderTrustworthy(mssg[1], agent)
                #         # mssg[2] = True
                #         self.receivedMessages[agent][i] = (mssg[0], mssg[1], True)

    def initTrustBeliefs(self):
        for member in self._teamMembers:
            self.trustBeliefs[member] = 0.5

    def acceptMessageIfSenderTrustworthy(self, mssg, sender):
        splitMssg = mssg.split(' ')
        if splitMssg[0] == 'Moving' and splitMssg[1] == 'to':
            room_to = splitMssg[2]
            if self.trustBeliefs[sender] >= 0.5:
                for room, door in self.rooms_to_visit:
                    if room_to == room:
                        self.rooms_to_visit.remove((room, door))
                        self.visited.append((room, door))
                        print("VRATAAAAAAAAAa")
                        print(door)
                        self.closed_doors.remove(door["room_name"])

        if splitMssg[0] == 'Opening' and splitMssg[1] == 'door':
            # TODO maybe we need to call verify_action_sequence first
            # if self.trustBeliefs[sender] >= 0.5:
            print("VRATATAAAAAAAAAAAAAA")
            print(splitMssg[3])
            print(self.closed_doors)
            if self.verify_action_sequence(self.receivedMessages, sender, self.closed_doors):
                print("OPAAAAAAAAa")
                # self.closed_doors.remove(splitMssg[3])
            # pass
            pass

        if splitMssg[0] == 'Searching' and splitMssg[1] == 'through':
            pass

        if splitMssg[0] == 'Found' and splitMssg[1] == 'goal':
            vis, loc = self.getVisLocFromMessage(mssg)
            if self.trustBeliefs[sender] >= 0.5:
                for obj_vis, dropoff_loc in self.all_desired_objects:
                    if self.compareObjects(vis, obj_vis):
                        self.addToMemory(vis, loc, dropoff_loc)

        if splitMssg[0] == 'Dropped' and splitMssg[1] == 'goal':
            if self.trustBeliefs[sender] >= 0.5:
                # self.dropped_off_count += 1
                pass

        if splitMssg[0] == 'Picking' and splitMssg[2] == 'goal':
            vis, loc = self.getVisLocFromMessage(mssg)
            if self.trustBeliefs[sender] >= 0.4:
                for dict1 in self.memory:
                    if self.compareObjects(dict1['visualization'], vis):
                        self.memory.remove(dict1)
                for obj in self.desired_objects:
                    if self.compareObjects(obj[0], vis):
                        self.desired_objects.remove(obj)

    def checkMessageTrue(self, ticks, mssg, sender):
        splitMssg = mssg.split(' ')
        if splitMssg[0] == 'Moving' and splitMssg[1] == 'to':
            pass

        if splitMssg[0] == 'Opening' and splitMssg[1] == 'door':
            pass

        if splitMssg[0] == 'Searching' and splitMssg[1] == 'through':
            pass

        if splitMssg[0] == 'Found' and splitMssg[1] == 'goal':
            vis, loc = self.getVisLocFromMessage(mssg)
            for obj in self.seenObjects:
                if self.compareObjects(vis, obj[0]) and obj[1] == loc:
                    return True
                elif str(obj[1]) == loc:
                    return False

        if splitMssg[0] == 'Dropped' and splitMssg[1] == 'goal':
            pass

        if splitMssg[0] == 'Picking' and splitMssg[2] == 'goal':
            pass

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
        keys = ('shape', 'colour')
        for key in keys:
            if key in obj1 and key in obj2:
                if obj1[key] != obj2[key]:
                    return False
        return True

    def addToSeenObjects(self, obj):
        if obj not in self.seenObjects:
            self.seenObjects.append(obj)

    def verify_action_sequence(self, mssgs, sender, closed_doors):
        mssg, prev_mssg = self.find_mssg(mssgs, sender)

        if prev_mssg is not None:
            prev = prev_mssg.split(' ')
            curr = mssg.split(' ')
            # check if all door are open when a message for opening a door is received
            # closed_doors = [door for door in state.values()
            #                 if 'class_inheritance' in door and 'Door' in door['class_inheritance'] and not door[
            #         'is_open']]
            if (prev[0] == 'Opening' or curr[0] == 'Opening') and len(closed_doors) == 0:
                print('Door is already open, dummy')
                return False

            if (prev[0] == 'Opening' and prev[3] not in closed_doors) or (
                    curr[0] == 'Opening' and curr[3] not in closed_doors):
                print("TUKAAAAAAAAAAAAaa")
                return False

            # check moving to room, opening door sequence
            if prev[0] == 'Moving':
                # decrease trust score by little is action after moving to a room is not opening a door -> Lazy agent
                if curr[0] != 'Opening':
                    print('Invalid action sequence')
                    return False

                # decrease trust score if an agent says that he is going to one room, but opening the door of another
                if curr[0] == 'Opening' and prev[2] != curr[3]:
                    print('That is another room, dummy')
                    return False
                elif curr[0] == 'Opening' and prev[2] == curr[3]:
                    return True

            if curr[0] == 'Searching':
                if prev[0] == 'Moving' and curr[2] == prev[2]:
                    return True
                else:
                    return False

            if curr[0] == 'Picking':
                if prev[0] == 'Found':
                    pass
                else:
                    return False
        return

    def find_mssg(self, mssgs, from_id):
        counter = 0
        mssg = None
        prev_mssg = None
        for mssg_i in mssgs[from_id]:
            if counter == 0:
                mssg = mssg_i[1]
                counter = counter + 1
            else:
                prev_mssg = mssg_i[1]
                break
        return mssg, prev_mssg