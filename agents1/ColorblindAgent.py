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

class ColorblindAgent(BW4TBrain):

    def __init__(self, settings: Dict[str, object]):
        super().__init__(settings)
        self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR
        self._teamMembers = []
        self.desired_objects = []
        # only the strong agents can pick 2 blocks
        # for other agents this is 0 or 1
        self.capacity = 0
        self.drop_off_locations = []
        self.object_to_be_dropped = None
        self.initialization_flag = True
        self.all_desired_objects = []

        # memory keeps track of the objects that were located but should be retrieved later
        #   it contains the following information
        #   {
        #       "visualization" : the visualization of the object that has to be picked up
        #       "location"      : the location where the object was found (TODO if a specific object is needed,
        #                           go to the nearest object with that visualization if multiple are available in this array)
        #       "drop_off_location" : where the object need to be dropped
        #   }
        self.memory = []
        self.all_rooms = []
        self.detected_objects = []
        self.processed_messages = []

        self.grab = False
        self.drop = False

        self.obj_id = None
        self.agent_name = None
        self.seenObjects = []
        self.ticks = 0
        self.receivedMessages = {}
        self.totalMessagesReceived = 0
        # A list of messages To Be Verified
        self.tbv = []
        # For each team member store trust score
        self.trustBeliefs = {}
        self.rooms_to_visit = []
        self.visited = []
        self._door = None
        self.dropped_off_count = 0
        self.obj_id = None
        self.closed_doors = []

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

        # We check if we enter for first time in the method as there is recursion
        # We want to keep track of some objects and reinitialize them every time
        if self.initialization_flag:

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
                found_obj.append(({"shape": obj["visualization"]["shape"], "colour": None }, obj["location"]))
            self.desired_objects = sorted(found_obj, key=lambda x: x[1], reverse=True)

            self.all_desired_objects = self.desired_objects.copy()
            sorted(self.all_desired_objects, key=lambda obj: obj[1], reverse=True)

        while True:
            # TODO parse all new messages
            # if a desired object is found, add it to self.detected_objects list
            # if an object from detected_objects has been collected/dropped, remove it from the list
            #   AND remove the last waypoint from the navigator
            #   AND self._phase = self.previous_phase
            #   AND keep track of already dropped objects
            for msg in self.received_messages:
                if not msg in self.processed_messages and msg.from_id != self.agent_id:
                    self._parseMessage(msg)
                    self.processed_messages.append(msg)

            if len(self.detected_objects) > 0:
                self.previous_phase = self._phase
                self._phase = Phase.FOLLOW_PATH_TO_DESIRED_OBJECT
                # TODO do something

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
                self._navigator.add_waypoints(sorted_by_xy)

                # Go to the next phase
                self._phase = Phase.TRAVERSE_ROOM

            if Phase.TRAVERSE_ROOM == self._phase:
                # Every time update the state for the new location of the agent
                self._state_tracker.update(state)

                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
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
                        for des, loc in self.desired_objects:
                            if obj[0]["shape"] == des["shape"]:
                                # In case they are desired objects for the strong agent we are interested only in the
                                # first two items from bottom to up, if they are we pick them
                                # in case they are not we save them in the memory for later use
                                if ((des, loc)) in self.desired_objects:
                                #        and \
                                #        not ((des, obj[2])) in map((lambda mem: (mem["visualization"], mem["location"])), self.memory):
                                # if ((des, loc)) != self.desired_objects[0] \
                                #         and ((des, loc)) in self.desired_objects:

                                    self._sendMessage("Found " + str(obj[0]["shape"]), self.agent_id)
                                    self.memory.append({ "visualization": { "shape": des["shape"], "colour": None }, "location": obj[2], "drop_off_location": loc })
                                    self.memory.sort(key= lambda mem: mem["location"], reverse=True)

                    # If no desired object was found just move
                    return action, {}

                # If the room is traversed go to te next room
                self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR

            # Find the path to the deliver location
            if Phase.DELIVER_ITEM == self._phase:
                locations = []
                # sort the location of the picked items so that the first dropped will be at the bottom
                for _, loc in self.drop_off_locations:
                    locations.append(loc)
                locations.sort(reverse=True)
                self._navigator.reset_full()
                # Add the navigation
                self._navigator.add_waypoints(locations)

                # Next phase
                self._phase = Phase.FOLLOW_PATH_TO_DROP_OFF_LOCATION

            # Follow path to the drop off location
            if Phase.FOLLOW_PATH_TO_DROP_OFF_LOCATION == self._phase:
                flag = False
                # Check if the current location of the agent is the correct drop off location
                for obj_id, loc in self.drop_off_locations:
                    if state[self._state_tracker.agent_id]['location'] == loc:
                        flag = True
                        self.object_to_be_dropped = obj_id
                        # if it is the correct location drop the object
                        self._phase = Phase.DROP_OBJECT
                        self.drop_off_locations.remove((obj_id, loc))

                # if not already dropped the object  move to the next location
                if not flag:
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
                        if len(self.memory) != 0:
                            self._navigator.reset_full()
                            self._navigator.add_waypoints([self.memory.peek()["location"]])
                            self._phase = Phase.TRAVERSE_ROOM
                        else:
                            # If memory is empty continue traversing rooms
                            self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR

                print("! DONE !")

            if Phase.DROP_OBJECT == self._phase:
                if self.object_to_be_dropped is None:
                    print("CODE BROKEN VERY BAD")
                    exit(-1)
                # update capacity
                self.capacity -= 1
                print("dropped object")
                # Drop object
                self._phase = Phase.FOLLOW_PATH_TO_DROP_OFF_LOCATION

                return DropObject.__name__, {'object_id': self.object_to_be_dropped}

            if Phase.PLAN_PATH_TO_CLOSED_DOOR == self._phase:
                self._navigator.reset_full()

                closedDoors = [door for door in state.values()
                               if 'class_inheritance' in door and 'Door' in door['class_inheritance'] and not door[
                        'is_open']]

                # Randomly pick a closed door or go to open room
                # Check if all rooms open
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
                    self._door = random.choice(closedDoors)

                # get the location of the door
                doorLoc = self._door['location']

                # Location in front of door is south from door
                doorLoc = doorLoc[0], doorLoc[1] + 1

                # Send message of current action
                self._sendMessage('Moving to door of ' + self._door['room_name'], agent_name)
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
                return OpenDoorAction.__name__, {'object_id': self._door['obj_id']}


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
                    self.obj_id = self.getObjectIdFromLocation(state,
                                                               state[self._state_tracker.agent_id]['location'])
                    self.grab = True
                    return GrabObject.__name__, {'object_id': self.obj_id}
                if not self.drop:
                    self.drop = True
                    return DropObject.__name__, {'object_id': self.obj_id}

                self.grab = False
                self.drop = False
                self._phase = Phase.REORDER_ITEMS

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

    def _parseMessage(self, msg):
        if "Found" in msg.content:
            message = msg.content.split()
            message = message[message.index("Found")+1:]
            print(message)

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
        for x in range(min_xy[0] + 1, max_xy[0]):
            for y in range(min_xy[1] + 1, max_xy[1] - 1):
                list_coordinates.append((x, y))
                # print(x, y)

        self._navigator.add_waypoints(list_coordinates)

    def getObjectIdFromLocation(self, state, loc):
        for obj in state.get_closest_with_property("is_collectable"):
            if obj["is_collectable"] is True and \
                    not 'GhostBlock' in obj['class_inheritance'] and obj["location"] == loc:
                return obj["obj_id"]

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

    # TODO CORRECT METHOD FOR PROCESSING MESSAGES IS BELOW
    # TODO CODE NEEDS TO BE FIXED IN ORDER TO USE  THIS METHOD
    # def _processMessages(self):
    #     '''
    #     Process incoming messages and create a dictionary with received messages from each team member.
    #     '''
    #
    #     for mssg in self.received_messages[self.totalMessagesReceived:]:
    #         for member in self._teamMembers:
    #             if mssg.from_id == member:
    #                 self.receivedMessages[member].append((self.ticks, mssg.content, False))
    #                 self.totalMessagesReceived = self.totalMessagesReceived + 1
    #                 self.tbv.append((self.ticks, mssg.content, mssg.from_id))
    #                 self.acceptMessageIfSenderTrustworthy(mssg.content, mssg.from_id)
    #     tbv_copy = self.tbv
    #     for (ticks, mssg, from_id) in tbv_copy:
    #         is_true = self.checkMessageTrue(self.ticks, mssg, from_id) or \
    #                   self.verify_action_sequence(self.receivedMessages, from_id, self.closed_doors)
    #         if is_true is not None:
    #             if is_true:
    #                 self.increaseTrust(from_id)
    #             else:
    #                 self.decreaseTrust(from_id)
    #             self.tbv.remove((ticks, mssg, from_id))

    def believeAgent(self):
        for agent in self.receivedMessages:
            if self.trustBeliefs[agent] >= 0.9:
                for i in range(len(self.receivedMessages[agent])):
                    mssg = self.receivedMessages[agent][i]
                    if not mssg[2]:
                        self.acceptMessageIfSenderTrustworthy(mssg[1], agent)
                        # mssg[2] = True
                        self.receivedMessages[agent][i] = (mssg[0], mssg[1], True)

    def initTrustBeliefs(self):
        for member in self._teamMembers:
            self.trustBeliefs[member] = 0.5

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

    def acceptMessageIfSenderTrustworthy(self, mssg, sender):
        splitMssg = mssg.split(' ')
        if splitMssg[0] == 'Moving' and splitMssg[1] == 'to':
            room_to = splitMssg[2]
            if self.trustBeliefs[sender] >= 0.5:
                for room, door in self.rooms_to_visit:
                    if room_to == room:
                        self.rooms_to_visit.remove((room, door))
                        self.visited.append((room, door))

        if splitMssg[0] == 'Opening' and splitMssg[1] == 'door':
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
                self.dropped_off_count += 1

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
            # check if all door are open when a message for opening a door is received
            # closed_doors = [door for door in state.values()
            #                 if 'class_inheritance' in door and 'Door' in door['class_inheritance'] and not door[
            #         'is_open']]
            if (prev[0] == 'Opening' or mssg.split(' ')[0] == 'Opening') and len(closed_doors) == 0:
                return False

            # check moving to room, opening door sequence
            if prev[0] == 'Moving':
                curr = mssg.split(' ')

                # decrease trust score by little is action after moving to a room is not opening a door -> Lazy agent
                # TODO check whether door is not open
                if curr[0] != 'Opening' and curr[2] not in closed_doors:
                    return False

                # decrease trust score if an agent says that he is going to one room, but opening the door of another
                if curr[0] == 'Opening' and prev[2] != curr[2]:
                    return False

            return True
        return False

    def find_mssg(self, mssgs, from_id):
        counter = 0
        mssg = None
        prev_mssg = None
        for mssg in mssgs:
            if mssg[2] == from_id:
                if (counter == 0):
                    mssg = mssg[1]
                    counter = counter + 1
                else:
                    prev_mssg = mssg[1]
                    break

        return mssg, prev_mssg