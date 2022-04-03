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
    PLAN_PATH_TO_CLOSED_DOOR = 1
    FOLLOW_PATH_TO_CLOSED_DOOR = 2
    OPEN_DOOR = 3
    ENTER_ROOM = 4
    TRAVERSE_ROOM = 5
    DELIVER_ITEM = 6
    FOLLOW_PATH_TO_DROP_OFF_LOCATION = 7
    DROP_OBJECT = 8
    GO_TO_REORDER_ITEMS = 9
    REORDER_ITEMS = 10
    GRAB_AND_DROP = 11
    CHECK_ITEMS = 12

# What the difference between the trust scores should be when agents are sharing their
# trust scores with the world in order to influence other agents' scores
EPSILON = 0.15

class StrongAgent(BW4TBrain):

    def __init__(self, settings: Dict[str, object]):
        super().__init__(settings)
        self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR
        self._teamMembers = []
        self.desired_objects = []
        self.all_desired_objects = []
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
        self.rooms_to_visit = []
        self.visited = []
        # used for the last phase - GRAB_AND_DROP to keep track of when an object is grabbed and after it was just dropped
        self.grab = False
        self.drop = False
        self._door = None
        self.dropped_off_count = 0
        self.at_drop_location = {}
        self.obj_id = None
        self.closed_doors = []
        self.not_dropped = []
        self.drop_counter = 0

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

        # share trust scored every 25th tick
        if self.ticks % 25 == 0:
            self.shareTrustScores()

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
                self.at_drop_location[obj["location"]] = 0

            self.desired_objects = sorted(found_obj, key=lambda x: x[1], reverse=True)

            self.initTrustBeliefs()
            self._init_trust_table(state['World']['team_members'])
            self.all_desired_objects = self.desired_objects.copy()
            sorted(self.all_desired_objects, key=lambda obj: obj[1], reverse=True)

            for room in self.all_rooms:
                if room != "world_bounds":
                    door = state.get_room_doors(room)
                    self.rooms_to_visit.append((room, door[0]))

            self.closed_doors = [door for door in state.values()
                            if 'class_inheritance' in door and 'Door' in door['class_inheritance'] and not door[
                    'is_open']]
            for i, door in enumerate(self.closed_doors):
                self.closed_doors[i] = self.closed_doors[i]["room_name"]
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

                drop = self.check_for_not_dropped()
                if drop is not None:
                    return drop

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
                check = False
                for obj in found_obj:
                    self.addToSeenObjects((obj[0], obj[2]))
                    for des, loc in self.desired_objects:
                        if obj[0]["shape"] == des["shape"] and obj[0]["colour"] == des["colour"]:
                            # In case they are desired objects for the strong agent we are interested only in the
                            # first two items from bottom to up, if they are we pick them
                            # in case they are not we save them in the memory for later use
                            check = True
                            self._messageFoundGoalBlock(str(obj[0]), str(obj[2]))

                            if (des, loc) in self.desired_objects:
                                if self.capacity < 2:
                                    self.capacity += 1
                                    self.drop_off_locations.append((obj[0], obj[1], loc))
                                    self.desired_objects.remove((des, loc))
                                    self._messagePickUpGoalBlock(str(obj[0]), str(obj[2]))

                                    for dict1 in self.memory:
                                        if obj[0]["shape"] == dict1["visualization"]["shape"] \
                                                and obj[0]["colour"] == dict1["visualization"]["colour"]:
                                            self.memory.remove(dict1)

                                    # Grab object if there is a capacity
                                    return GrabObject.__name__, {'object_id': obj[1]}
                                else:
                                    self.addToMemory(obj[0], obj[2], loc)
                    if not check:
                        self._messageFoundBlock(str(obj[0]), str(obj[2]))
                        check = False
                # In case we are filled, deliver items, next phase
                if self.capacity > 1:
                    self._phase = Phase.DELIVER_ITEM
                # In case there is only one object left needed and is found deliver it, next phase
                elif len(self.desired_objects) == 0 and self.capacity > 0:
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
                self._navigator.add_waypoints(locations)

                # Next phase
                self._phase = Phase.FOLLOW_PATH_TO_DROP_OFF_LOCATION

            # Follow path to the drop off location
            if Phase.FOLLOW_PATH_TO_DROP_OFF_LOCATION == self._phase:
                flag = False
                flag_not_dropped = False
                self.object_to_be_dropped = None
                for obj_viz, obj_id, loc in self.drop_off_locations:
                    if state[self._state_tracker.agent_id]['location'] == loc:
                        flag = True
                        # if it is the correct location drop the object
                        self.drop_off_locations.remove((obj_viz, obj_id, loc))

                        for obj in state.get_closest_with_property("is_collectable"):
                            if obj["is_collectable"] is True and not 'GhostBlock' in obj['class_inheritance'] and obj["location"] == loc:
                                self.not_dropped.append((obj_id, loc))
                                flag_not_dropped = True
                                self.object_to_be_dropped = None
                        if not flag_not_dropped:
                            self.object_to_be_dropped = (obj_id, loc)
                            self._messageDroppedGoalBlock(str(obj_viz), str(loc))
                            self._phase = Phase.DROP_OBJECT


                # if not already dropped the object move to the next location
                if not flag:
                    self._state_tracker.update(state)

                    action = self._navigator.get_move_action(self._state_tracker)
                    # Move to the next location
                    if action != None:
                        return action, {}
                    else:
                        # Go check if items are ordered
                        locations = list(map(lambda des_obj: des_obj[1], self.all_desired_objects))
                        self._navigator.reset_full()
                        # Add the navigation
                        self._navigator.add_waypoints(locations)
                        self.dropped_off_count = 0
                        self._phase = Phase.CHECK_ITEMS

            if Phase.DROP_OBJECT == self._phase:
                if self.object_to_be_dropped is None:
                    self._phase = Phase.FOLLOW_PATH_TO_DROP_OFF_LOCATION

                # update capacity
                else:
                    self.capacity -= 1

                    # Drop object
                    self._phase = Phase.FOLLOW_PATH_TO_DROP_OFF_LOCATION

                    return DropObject.__name__, {'object_id': self.object_to_be_dropped[0]}

            if Phase.PLAN_PATH_TO_CLOSED_DOOR == self._phase:
                self._navigator.reset_full()

                doc = 0
                for key in self.at_drop_location:
                    doc += self.at_drop_location[key]
                if doc == len(self.all_desired_objects):
                    self._phase = Phase.GO_TO_REORDER_ITEMS
                    return None, {}

                if len(self.memory) > 0:
                    self._navigator.reset_full()
                    self._navigator.add_waypoints([self.memory[0]["location"]])

                    if len(self.not_dropped) > 0:
                        self.dropped_off_count = self.shortestDistance_drop(state, self.memory[0]["location"])
                    else:
                        self.dropped_off_count = -1
                    self.memory.pop(0)
                    self._phase = Phase.TRAVERSE_ROOM
                # Randomly pick a closed door or go to open room
                # Check if all rooms open
                else:
                    if len(self.rooms_to_visit) != 0:
                        self._door = self.rooms_to_visit.pop(random.randint(0, len(self.rooms_to_visit) - 1))[1]
                    elif len(self.visited) != 0:
                        self._door = self.visited.pop()[1]
                    else:
                        if len(self.all_rooms) == 0:
                            return None, {}
                        # get the first room, as they were sorted in the first iteration
                        room_name = self.all_rooms.pop(0)
                        self.all_rooms.append(room_name)
                        # get the door of the chosen room
                        self._door = [loc for loc in state.values()
                                      if "room_name" in loc and loc['room_name'] is
                                      room_name and 'class_inheritance' in loc and
                                      'Door' in loc['class_inheritance']][0]

                        # in case some broken room without door - stuck
                        if len(self._door) == 0:
                            return None, {}

                    doorLoc = self._door["location"]
                    # Location in front of door is south from door
                    doorLoc = (doorLoc[0], doorLoc[1] + 1)

                    # Send message of current action
                    self._messageMoveRoom(self._door['room_name'])
                    self._navigator.add_waypoints([doorLoc])

                    if len(self.not_dropped) > 0:
                        self.dropped_off_count = self.shortestDistance_drop(state, doorLoc)
                    else:
                        self.dropped_off_count = -1
                    # go to the next phase
                    self._phase = Phase.FOLLOW_PATH_TO_CLOSED_DOOR

            if Phase.FOLLOW_PATH_TO_CLOSED_DOOR == self._phase:
                self._state_tracker.update(state)
                # Follow path to door

                drop = self.check_for_not_dropped()
                if drop is not None:
                    return drop
                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    return action, {}
                # go to the next phase
                self._phase = Phase.OPEN_DOOR

            if Phase.OPEN_DOOR == self._phase:
                self._phase = Phase.ENTER_ROOM
                # Open door
                # If already opened, no change
                if self._door['room_name'] in self.closed_doors:
                    self._messageOpenDoor(self._door['room_name'])
                return OpenDoorAction.__name__, {'object_id': self._door['obj_id']}

            if Phase.GO_TO_REORDER_ITEMS == self._phase:
                # sort the location of the picked items so that the first dropped will be at the bottom
                locations = list(map(lambda des_obj: des_obj[1], self.all_desired_objects))
                self._navigator.reset_full()
                # Add the navigation
                self._navigator.add_waypoints(locations)

                self._phase = Phase.REORDER_ITEMS

            if Phase.REORDER_ITEMS == self._phase:
                if len(self.all_desired_objects) != 0:
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
                            # the simulation should be done
                elif len(self.all_desired_objects) <= 0:
                    self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR

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
                # Every time update the state for the new location of the agent
                self._state_tracker.update(state)

                action = self._navigator.get_move_action(self._state_tracker)

                found = False
                myLoc = state[self._state_tracker.agent_id]['location']
                for des, loc in self.all_desired_objects:
                    if myLoc == loc:
                        for obj in state.get_closest_with_property("is_collectable"):
                            if obj["is_collectable"] is True and not 'GhostBlock' in obj['class_inheritance'] and obj[
                                "location"] == myLoc:
                                if self.compareObjects(des, obj):
                                    self.at_drop_location[loc] = 1
                                    self._messageFoundBlock(str(des), str(loc))
                                    found = True
                                    break
                        else:
                            continue
                        break

                if not found:
                    for des, loc in self.all_desired_objects:
                        if myLoc == loc:
                            if (des, loc) not in self.desired_objects:
                                self.desired_objects.append((des, loc))
                if action is not None:
                    return action, {}
                else:
                    self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR

    def getLength(self):
        return 0.9

    def shortestDistance_drop(self, state, go_to):
        current_loc = state[self._state_tracker.agent_id]['location']
        if current_loc[0] > go_to[0]:
            x = current_loc[0] - go_to[0]
        else:
            x = go_to[0] - current_loc[0]

        if current_loc[1] > go_to[1]:
            y = current_loc[1] - go_to[1]
        else:
            y = go_to[1] - current_loc[1]

        distance = (x ** 2 + y ** 2) ** 0.5

        return int(round(distance * self.getLength()))

    def check_for_not_dropped(self):
        if self.dropped_off_count > 0:
            self.dropped_off_count -= 1
        elif self.dropped_off_count == 0:
            if len(self.not_dropped) > 0:
                if self.capacity > 0:
                    self.capacity -=1

                item = self.not_dropped.pop(0)[0]

                return DropObject.__name__, {'object_id': item}

    def getObjectIdFromLocation(self, state, loc):
        for obj in state.get_closest_with_property("is_collectable"):
            if obj["is_collectable"] is True and not 'GhostBlock' in obj['class_inheritance'] and obj["location"] == loc:
                return obj["obj_id"]
        return

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
        df.to_csv("Trust.csv")

    def _write_to_trust_table(self, trustor, trustee, new_trust):
        df = pd.read_csv('Trust.csv', index_col=0)
        df.loc[trustor, trustee] = new_trust
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
                    is_sequence_true = self.verify_action_sequence(self.receivedMessages, member)
                    if is_sequence_true is not None:
                        if is_sequence_true:
                            self.increaseTrust(member)
                        else:
                            self.decreaseTrust(member)
            self.already_said(mssg.content, mssg.from_id)
        tbv_copy = self.tbv
        for (ticks, mssg, from_id) in tbv_copy:
            is_true = self.checkMessageTrue(self.ticks, mssg, from_id)
            if is_true is not None:
                if is_true:
                    self.increaseTrust(from_id)
                else:
                    self.decreaseTrust(from_id)
                self.tbv.remove((ticks, mssg, from_id))

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
            if self.trustBeliefs[sender] >= 0.6:
                for room, door in self.rooms_to_visit:
                    if room_to == room:
                        self.rooms_to_visit.remove((room, door))
                        self.visited.append((room, door))
                        # remove door of room from all closed_doors
                        self.closed_doors.remove(door["room_name"])

        if splitMssg[0] == 'Opening' and splitMssg[1] == 'door':
            pass

        if splitMssg[0] == 'Searching' and splitMssg[1] == 'through':
            pass

        if splitMssg[0] == 'Found' and splitMssg[1] == 'goal':
            vis, loc = self.getVisLocFromMessage(mssg)
            if self.trustBeliefs[sender] >= 0.6:
                for obj_vis, dropoff_loc in self.all_desired_objects:
                    if self.compareObjects(vis, obj_vis):
                        self.addToMemory(vis, loc, dropoff_loc)

        if splitMssg[0] == 'Dropped' and splitMssg[1] == 'goal':
            if self.trustBeliefs[sender] >= 0.6:
                pass

        if splitMssg[0] == 'Picking' and splitMssg[2] == 'goal':
            vis, loc = self.getVisLocFromMessage(mssg)
            if self.trustBeliefs[sender] >= 0.6:
                for dict1 in self.memory:
                    if self.compareObjects(dict1['visualization'], vis):
                        self.memory.remove(dict1)
                for obj in self.desired_objects:
                    if self.compareObjects(obj[0], vis):
                        self.desired_objects.remove(obj)


        if mssg.startswith("Trust score of "):
            agent, trust_score = self.getAgentScoreFromMessage(mssg)
            # sender will not send a trust score about itself, but check this is true just in case
            if trust_score is not None and self.trustBeliefs[sender] >= 0.6 and agent != self.agent_name and sender != agent: # TODO 0.6
                if self.trustBeliefs[agent] < trust_score - EPSILON:
                    self.trustBeliefs[agent] = np.clip(self.trustBeliefs[agent] + 0.05, 0, 1)
                    self._write_to_trust_table(self.agent_id, agent, self.trustBeliefs[agent])
                elif self.trustBeliefs[agent] > trust_score + EPSILON:
                    self.trustBeliefs[agent] = np.clip(self.trustBeliefs[agent] - 0.05, 0, 1)
                    self._write_to_trust_table(self.agent_id, agent, self.trustBeliefs[agent])
                # else don't do anything

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

    def getAgentScoreFromMessage(self, mssg):
        mssg_split = mssg.split()
        trust_score = None
        try:
            trust_score = float(mssg_split[5])
        except ValueError:
            print("ERROR when parsing message about trust. Incorrect message format: trust is not a float.")

        return mssg_split[3], trust_score

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
        flag = True
        for key in keys:
            if key in obj1 and key in obj2:
                if obj1[key] == obj2[key] or obj1[key] is None or obj2[key] is None:
                    pass
                else:
                    flag = False
        return flag

    def addToSeenObjects(self, obj):
        if obj not in self.seenObjects:
            self.seenObjects.append(obj)

    def shareTrustScores(self):
        for agent in self.trustBeliefs:
            if agent != self.agent_name:
                belief = self.trustBeliefs[agent]
                self._sendMessage("Trust score of " + agent + " is " + str(belief), self.agent_name)

    def verify_action_sequence(self, mssgs, sender):
        mssg, prev_mssg = self.find_mssg(mssgs, sender)

        if prev_mssg is not None:
            prev = prev_mssg.split(' ')
            curr = mssg.split(' ')
            # check if all door are open when a message for opening a door is received
            if (prev[0] == 'Opening' or curr[0] == 'Opening') and len(self.closed_doors) == 0:
                return False

            if (prev[0] == 'Opening' and prev[3] not in self.closed_doors) or (curr[0] == 'Opening' and curr[3] not in self.closed_doors):
                return False

            # check moving to room, opening door sequence
            if prev[0] == 'Moving':
                # decrease trust score by little is action after moving to a room is not opening a door -> Lazy agent
                if curr[0] != 'Opening':
                    return False

                # decrease trust score if an agent says that he is going to one room, but opening the door of another
                if curr[0] == 'Opening' and prev[2] != curr[3]:
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
            # ignore messages about the trust score of an agent
            if not "Trust score of " in mssg_i[1]:
                if counter == 0:
                    mssg = mssg_i[1]
                    counter = counter + 1
                else:
                    prev_mssg = mssg_i[1]
                    break
        return mssg, prev_mssg


    def already_said(self, received_mssg, sender):
        for name in self.receivedMessages.keys():
            if name != self.agent_id and name != sender:
                for mssg in self.receivedMessages[name]:
                    vis, loc = self.getVisLocFromMessage(mssg[1])
                    received_vis, received_loc = self.getVisLocFromMessage(received_mssg[1])
                    if mssg[1].split(' ')[0] == 'Found':
                        if received_mssg[1].split(' ')[0] == 'Found':
                            if self.compareObjects(vis, received_vis) and loc == received_loc:
                                self.increaseTrust(sender)
                                self.increaseTrust(name)
                        elif received_mssg[1].split(' ')[0] == 'Picking':
                            if self.compareObjects(vis, received_vis) and loc == received_loc:
                                self.increaseTrust(sender)
                                self.increaseTrust(name)
                        elif received_mssg[1].split(' ')[0] == 'Dropped':
                            if self.compareObjects(vis, received_vis) and loc == received_loc:
                                self.increaseTrust(sender)
                                self.increaseTrust(name)

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

        self.not_dropped = []
        self.drop_counter = 0

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

        # share trust scored every 25th tick
        if self.ticks % 25 == 0:
            self.shareTrustScores()

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
                    door = state.get_room_doors(room)
                    self.rooms_to_visit.append((room, door[0]))
                    self.all_rooms.append(room)

            self.closed_doors = [door for door in state.values()
                            if 'class_inheritance' in door and 'Door' in door['class_inheritance'] and not door[
                    'is_open']]
            for i, door in enumerate(self.closed_doors):
                self.closed_doors[i] = self.closed_doors[i]["room_name"]

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

                room = self._door['room_name']
                self._messageSearchThrough(room)
                if decision:
                    self._navigator.add_waypoints(sorted_by_xy)
                    self._phase = Phase.TRAVERSE_ROOM
                    self.stop_when = -1
                    if self.checked_room in self.rooms_to_visit:
                        self.rooms_to_visit.remove(self.checked_room)
                else:
                    self.stop_when = int(round(len(sorted_by_xy) * self.getRandom1()))
                    self._navigator.add_waypoints(sorted_by_xy)
                    self._phase = Phase.TRAVERSE_ROOM
                # Go to the next phase

            if Phase.TRAVERSE_ROOM == self._phase:
                # Every time update the state for the new location of the agent

                self._state_tracker.update(state)
                drop = self.check_for_not_dropped()
                if drop is not None:
                    return drop
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

                                if ((des, loc)) in self.desired_objects:
                                    decision = self.getRandom50()

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
                flag_not_dropped = False
                self.delivered_item = False
                self.object_to_be_dropped = None

                # Check if the current location of the agent is the correct drop off location

                if self.stop_when == 0:
                    self._phase = Phase.DROP_OBJECT
                    self._messageDroppedGoalBlock(str(self.my_object[0]),
                                                  str(state[self._state_tracker.agent_id]['location']))
                    self.addToMemory(self.my_object[0], state[self._state_tracker.agent_id]['location'],
                                     self.my_object[1])

                    self.object_to_be_dropped = self.my_object_id

                elif state[self._state_tracker.agent_id]['location'] == self.drop_off_locations[2]:
                    flag = True

                    self.object_to_be_dropped = None
                    self.delivered_item = True

                    # if it is the correct location drop the object
                    self._phase = Phase.DROP_OBJECT
                    if (self.my_object[0], self.my_object[1]) in self.desired_objects:
                        self.desired_objects.remove((self.my_object[0], self.my_object[1]))


                    for obj in state.get_closest_with_property("is_collectable"):
                        if obj["is_collectable"] is True and not 'GhostBlock' in obj['class_inheritance'] and obj[
                            "location"] == self.drop_off_locations[2]:
                            self.not_dropped.append((self.object_to_be_dropped, self.drop_off_locations[2]))
                            flag_not_dropped = True
                            self.object_to_be_dropped = None

                    if not flag_not_dropped:
                        self.object_to_be_dropped = self.drop_off_locations[1]
                        self._messageDroppedGoalBlock(str(self.drop_off_locations[0]), str(self.drop_off_locations[2]))
                        self._phase = Phase.DROP_OBJECT

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
                    locations = list(map(lambda des_obj: des_obj[1], self.all_desired_objects))
                    self._navigator.reset_full()
                    # Add the navigation
                    self._navigator.add_waypoints(locations)
                    self.dropped_off_count = 0
                    self._phase = Phase.CHECK_ITEMS

                # update capacity
                else:
                    self.capacity -= 1
                    # Drop object

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

                return DropObject.__name__, {'object_id': self.object_to_be_dropped}

            if Phase.PLAN_PATH_TO_CLOSED_DOOR == self._phase:
                self._navigator.reset_full()
                decision = self.getRandom50()
                doc = 0
                for key in self.at_drop_location:
                    doc += self.at_drop_location[key]
                if doc == len(self.all_desired_objects):
                    self._phase = Phase.GO_TO_REORDER_ITEMS
                    return None, {}


                if len(self.memory) > 0 and self.use_memory:

                    self._navigator.reset_full()
                    self._navigator.add_waypoints([self.memory[0]["location"]])
                    if len(self.not_dropped) > 0:
                        self.dropped_off_count = self.shortestDistance_drop(state, self.memory[0]["location"])
                    else:
                        self.dropped_off_count = -1
                    if decision:
                        self.stop_when = -1
                        self.use_memory = True
                        self.memory.pop(0)
                    else:
                        distance = self.getRandom1() * self.shortestDistance(
                            state[self._state_tracker.agent_id]['location'], self.memory[0]["location"])

                        self.stop_when = int(round(distance * self.getRandom1()))
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
                        self._door = [loc for loc in state.values()
                                      if "room_name" in loc and loc['room_name'] is
                                      room_name and 'class_inheritance' in loc and
                                      'Door' in loc['class_inheritance']][0]

                        # in case some broken room without door - stuck
                        if len(self._door) == 0:
                            return None, {}

                    doorLoc = self._door["location"]
                    # Location in front of door is south from door
                    doorLoc = (doorLoc[0], doorLoc[1] + 1)

                    if len(self.not_dropped) > 0:
                        self.dropped_off_count = self.shortestDistance_drop(state, doorLoc)
                    else:
                        self.dropped_off_count = -1
                    # Send message of current action
                    self._messageMoveRoom(self._door['room_name'])
                    self._navigator.add_waypoints([doorLoc])
                    # go to the next phase
                    self._phase = Phase.FOLLOW_PATH_TO_CLOSED_DOOR

                    decision = self.getRandom50()

                    if decision:
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
                drop = self.check_for_not_dropped()
                if drop is not None:
                    return drop
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

                decision = self.getRandom50()

                if decision:
                    if self._door['room_name'] in self.closed_doors:
                        self._messageOpenDoor(self._door['room_name'])
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
                if len(self.all_desired_objects) != 0:
                    if state[self._state_tracker.agent_id]['location'] == self.all_desired_objects[0][1]:
                        self.all_desired_objects.pop(0)
                        self._phase = Phase.GRAB_AND_DROP

                    if self._phase != Phase.GRAB_AND_DROP:
                        self._state_tracker.update(state)
                        action = self._navigator.get_move_action(self._state_tracker)
                        # Move to the next location
                        if action != None:
                            return action, {}
                        # otherwise the simulation should be done

                elif len(self.all_desired_objects) <= 0:
                    self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR

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
                # Every time update the state for the new location of the agent
                self._state_tracker.update(state)

                action = self._navigator.get_move_action(self._state_tracker)

                found = False
                myLoc = state[self._state_tracker.agent_id]['location']
                for des, loc in self.all_desired_objects:
                    if myLoc == loc:
                        for obj in state.get_closest_with_property("is_collectable"):
                            if obj["is_collectable"] is True and not 'GhostBlock' in obj['class_inheritance'] and obj[
                                "location"] == myLoc:
                                if self.compareObjects(des, obj):
                                    self._messageFoundBlock(str(des), str(loc))
                                    self.at_drop_location[loc] = 1
                                    break
                        else:
                            continue
                        break

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

    def shortestDistance_drop(self, state, go_to):
        current_loc = state[self._state_tracker.agent_id]['location']
        if current_loc[0] > go_to[0]:
            x = current_loc[0] - go_to[0]
        else:
            x = go_to[0] - current_loc[0]

        if current_loc[1] > go_to[1]:
            y = current_loc[1] - go_to[1]
        else:
            y = go_to[1] - current_loc[1]

        distance = (x ** 2 + y ** 2) ** 0.5

        return int(round(distance * 0.9))

    def check_for_not_dropped(self):
        if self.dropped_off_count > 0:
            self.dropped_off_count -= 1
        elif self.dropped_off_count == 0:
            if len(self.not_dropped) > 0:
                if self.capacity > 0:
                    self.capacity -=1
                item = self.not_dropped.pop(0)[0]
                return DropObject.__name__, {'object_id': item}

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
        df.to_csv("Trust.csv")

    def _write_to_trust_table(self, trustor, trustee, new_trust):
        df = pd.read_csv('Trust.csv', index_col=0)
        df.loc[trustor, trustee] = new_trust
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
            self.already_said(mssg.content, mssg.from_id)
        tbv_copy = self.tbv
        for (ticks, mssg, from_id) in tbv_copy:
            is_true = self.checkMessageTrue(self.ticks, mssg, from_id)
            if is_true is not None:
                if is_true:
                    self.increaseTrust(from_id)
                else:
                    self.decreaseTrust(from_id)
                self.tbv.remove((ticks, mssg, from_id))

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
                        self.closed_doors.remove(door["room_name"])

        if splitMssg[0] == 'Opening' and splitMssg[1] == 'door':
            pass

        if splitMssg[0] == 'Searching' and splitMssg[1] == 'through':
            pass

        if splitMssg[0] == 'Found' and splitMssg[1] == 'goal':
            vis, loc = self.getVisLocFromMessage(mssg)
            if self.trustBeliefs[sender] >= 0.6:
                for obj_vis, dropoff_loc in self.all_desired_objects:
                    if self.compareObjects(vis, obj_vis):
                        self.addToMemory(vis, loc, dropoff_loc)

        if splitMssg[0] == 'Dropped' and splitMssg[1] == 'goal':
            if self.trustBeliefs[sender] >= 0.6:
                pass

        if splitMssg[0] == 'Picking' and splitMssg[2] == 'goal':
            vis, loc = self.getVisLocFromMessage(mssg)
            if self.trustBeliefs[sender] >= 0.6:
                for dict1 in self.memory:
                    if self.compareObjects(dict1['visualization'], vis):
                        self.memory.remove(dict1)
                for obj in self.desired_objects:
                    if self.compareObjects(obj[0], vis):
                        self.desired_objects.remove(obj)

        if mssg.startswith("Trust score of "):
            agent, trust_score = self.getAgentScoreFromMessage(mssg)
            # sender will not send a trust score about itself, but check this is true just in case
            if trust_score is not None and self.trustBeliefs[sender] >= 0.6 and agent != self.agent_name and sender != agent: # TODO 0.6
                if self.trustBeliefs[agent] < trust_score - EPSILON:
                    self.trustBeliefs[agent] = np.clip(self.trustBeliefs[agent] + 0.05, 0, 1)
                    self._write_to_trust_table(self.agent_id, agent, self.trustBeliefs[agent])
                elif self.trustBeliefs[agent] > trust_score + EPSILON:
                    self.trustBeliefs[agent] = np.clip(self.trustBeliefs[agent] - 0.05, 0, 1)
                    self._write_to_trust_table(self.agent_id, agent, self.trustBeliefs[agent])
                # else don't do anything

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

    def getAgentScoreFromMessage(self, mssg):
        mssg_split = mssg.split()
        trust_score = None
        try:
            trust_score = float(mssg_split[5])
        except ValueError:
            print("ERROR when parsing message about trust. Incorrect message format: trust is not a float.")

        return mssg_split[3], trust_score

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
        flag = True
        for key in keys:
            if key in obj1 and key in obj2:
                if obj1[key] == obj2[key] or obj1[key] is None or obj2[key] is None:
                    pass
                else:
                    flag = False
        return flag

    def addToSeenObjects(self, obj):
        if obj not in self.seenObjects:
            self.seenObjects.append(obj)

    def shareTrustScores(self):
        for agent in self.trustBeliefs:
            if agent != self.agent_name:
                belief = self.trustBeliefs[agent]
                self._sendMessage("Trust score of " + agent + " is " + str(belief), self.agent_name)

    def verify_action_sequence(self, mssgs, sender, closed_doors):
        mssg, prev_mssg = self.find_mssg(mssgs, sender)

        if prev_mssg is not None:
            prev = prev_mssg.split(' ')
            curr = mssg.split(' ')
            # check if all door are open when a message for opening a door is received
            if (prev[0] == 'Opening' or curr[0] == 'Opening') and len(closed_doors) == 0:
                return False

            # check moving to room, opening door sequence
            if prev[0] == 'Moving':
                # decrease trust score by little is action after moving to a room is not opening a door -> Lazy agent
                if curr[0] != 'Opening':
                    return False

                # decrease trust score if an agent says that he is going to one room, but opening the door of another
                if curr[0] == 'Opening' and prev[2] != curr[3]:
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
            # ignore messages about the trust score of an agent
            if not "Trust score of " in mssg_i[1]:
                if counter == 0:
                    mssg = mssg_i[1]
                    counter = counter + 1
                else:
                    prev_mssg = mssg_i[1]
                    break
        return mssg, prev_mssg

    def already_said(self, received_mssg, sender):
        for name in self.receivedMessages.keys():
            if name != self.agent_id and name != sender:
                for mssg in self.receivedMessages[name]:
                    vis, loc = self.getVisLocFromMessage(mssg[1])
                    received_vis, received_loc = self.getVisLocFromMessage(received_mssg[1])
                    if mssg[1].split(' ')[0] == 'Found':
                        if received_mssg[1].split(' ')[0] == 'Found':
                            if self.compareObjects(vis, received_vis) and loc == received_loc:
                                self.increaseTrust(sender)
                                self.increaseTrust(name)
                        elif received_mssg[1].split(' ')[0] == 'Picking':
                            if self.compareObjects(vis, received_vis) and loc == received_loc:
                                self.increaseTrust(sender)
                                self.increaseTrust(name)
                        elif received_mssg[1].split(' ')[0] == 'Dropped':
                            if self.compareObjects(vis, received_vis) and loc == received_loc:
                                self.increaseTrust(sender)
                                self.increaseTrust(name)


class PossibleActions(enum.Enum):
    MOVING_TO_ROOM = 1
    OPENING_DOOR = 2
    SEARCHING_A_ROOM = 3
    ENCOUNTERING_A_GOAL_BLOCK = 4
    ENCOUNTERING_A_BLOCK = 5
    PICKING_UP_A_BLOCK = 6
    DROPPING_A_BLOCK = 7


class LiarAgent(BW4TBrain):

    def __init__(self, settings: Dict[str, object]):
        super().__init__(settings)
        self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR
        self._teamMembers = []
        # This list only keeps track of the remaining desired_objects
        self.desired_objects = []
        # this list keeps track of all desired objects and is not updated throughout the execution of the simulation
        self.all_desired_objects = []
        self.agent_name = None
        # only the strong agents can pick 2 blocks
        # for other agents this is 0 or 1
        self.capacity = 0
        self.drop_off_locations = []
        self.object_to_be_dropped = None
        self.initialization_flag = True
        self.memory = []
        self.all_rooms = []

        self.ticks = 0
        self.receivedMessages = {}
        self.totalMessagesReceived = 0
        # A list of messages To Be Verified
        self.tbv = []
        # For each team member store trust score
        self.trustBeliefs = {}
        self.rooms_to_visit = []
        self.visited = []
        self.seenObjects = []
        self.dropped_off_count = 0
        self.at_drop_location = {}
        # used for the last phase - GRAB_AND_DROP to keep track of when an object is grabbed and after it was just dropped

        self.grab = False
        self.drop = False

        self.obj_id = None
        self.closed_doors = []

        self.not_dropped = []
        self.drop_counter = 0

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

        # share trust scored every 25th tick
        if self.ticks % 25 == 0:
            self.shareTrustScores()

        # We check if we enter for first time in the method as there is recursion
        # We want to keep track of some objects and reinitialize them every time
        if self.initialization_flag:
            agent_name = state[self.agent_id]['obj_id']
            self.agent_name = agent_name

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
                self.at_drop_location[obj["location"]] = 0
            self.desired_objects = sorted(found_obj, key=lambda x: x[1], reverse=True)

            self.all_desired_objects = self.desired_objects.copy()
            sorted(self.all_desired_objects, key=lambda obj: obj[1], reverse=True)

            self.initTrustBeliefs()
            self._init_trust_table(state['World']['team_members'])

            for room in self.all_rooms:
                if room != "world_bounds":
                    door = state.get_room_doors(room)
                    self.rooms_to_visit.append((room, door[0]))

            self.closed_doors = [door for door in state.values()
                            if 'class_inheritance' in door and 'Door' in door['class_inheritance'] and not door[
                    'is_open']]
            for i, door in enumerate(self.closed_doors):
                self.closed_doors[i] = self.closed_doors[i]["room_name"]
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
                room = self._door['room_name']
                possible_action = self.pickAnAction(PossibleActions.SEARCHING_A_ROOM)
                if possible_action == PossibleActions.SEARCHING_A_ROOM:
                    # be honest
                    self._messageSearchThrough(room)
                else:
                    self._sendMessage(self.generateAMessageFromAction(possible_action), self.agent_name)

                # Add the locations of the tiles to traverse in order to the navigator
                self._navigator.reset_full()
                self._navigator.add_waypoints(sorted_by_xy)

                # Go to the next phase
                self._phase = Phase.TRAVERSE_ROOM

            if Phase.TRAVERSE_ROOM == self._phase:
                # Every time update the state for the new location of the agent
                self._state_tracker.update(state)
                drop = self.check_for_not_dropped()
                if drop is not None:
                    return drop

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
                check = False

                for obj in found_obj:
                    self.addToSeenObjects((obj[0], obj[2]))

                    for des, loc in self.desired_objects:
                        if obj[0]["shape"] == des["shape"] and obj[0]["colour"] == des["colour"]:
                            # In case they are desired objects for the strong agent we are interested only in the
                            # first two items from bottom to up, if they are we pick them
                            # in case they are not we save them in the memory for later use
                            possible_action = self.pickAnAction(PossibleActions.ENCOUNTERING_A_BLOCK)
                            if possible_action == PossibleActions.ENCOUNTERING_A_GOAL_BLOCK:
                                self._messageFoundGoalBlock(str(obj[0]), str(obj[2]))
                            else:
                                self._sendMessage(self.generateAMessageFromAction(possible_action), self.agent_name)

                            if ((des, loc)) in self.desired_objects:
                                if self.capacity == 0:
                                    self.capacity += 1
                                    self.drop_off_locations.append((obj[0], obj[1], loc))
                                    self.desired_objects.remove((des, loc))
                                    possible_action = self.pickAnAction(PossibleActions.PICKING_UP_A_BLOCK)
                                    if possible_action == PossibleActions.PICKING_UP_A_BLOCK:
                                        # be honest
                                        self._messagePickUpGoalBlock(str(obj[0]), str(obj[2]))
                                    else:
                                        self._sendMessage(self.generateAMessageFromAction(possible_action), self.agent_name)

                                    for dict1 in self.memory:
                                        if obj[0]["shape"] == dict1["visualization"]["shape"] \
                                                and obj[0]["colour"] == dict1["visualization"]["colour"]:
                                            self.memory.remove(dict1)

                                    return GrabObject.__name__, {'object_id': obj[1]}
                                else:
                                    self.addToMemory(obj[0], obj[2], loc)
                    if not check:
                        possible_action = self.pickAnAction(PossibleActions.ENCOUNTERING_A_BLOCK)
                        if possible_action == PossibleActions.ENCOUNTERING_A_BLOCK:
                            self._messageFoundBlock(str(obj[0]), str(obj[2]))
                        else:
                            self._sendMessage(self.generateAMessageFromAction(possible_action), self.agent_name)
                        check = False

                # In case we are filled, deliver items, next phase
                if self.capacity == 1:
                    self._phase = Phase.DELIVER_ITEM
                # In case there is only one object left needed and is found deliver it, next phase
                # If no desired object was found just move

                if action != None:
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
                self._navigator.add_waypoints(locations)

                # Next phase
                self._phase = Phase.FOLLOW_PATH_TO_DROP_OFF_LOCATION

            # Follow path to the drop off location
            if Phase.FOLLOW_PATH_TO_DROP_OFF_LOCATION == self._phase:
                flag = False
                flag_not_dropped = False
                self.object_to_be_dropped = None
                # Check if the current location of the agent is the correct drop off location
                for obj_viz, obj_id, loc in self.drop_off_locations:
                    if state[self._state_tracker.agent_id]['location'] == loc:
                        flag = True
                        self.object_to_be_dropped = obj_id
                        # if it is the correct location drop the object
                        self._phase = Phase.DROP_OBJECT
                        self.drop_off_locations.remove((obj_viz, obj_id, loc))
                        possible_action = self.pickAnAction(PossibleActions.DROPPING_A_BLOCK)

                        for obj in state.get_closest_with_property("is_collectable"):
                            if obj["is_collectable"] is True and not 'GhostBlock' in obj['class_inheritance'] and obj[
                                "location"] == loc:
                                self.not_dropped.append((obj_id, loc))
                                flag_not_dropped = True
                                self.object_to_be_dropped = None

                        if not flag_not_dropped:
                            self.object_to_be_dropped = obj_id
                            self._messageDroppedGoalBlock(str(obj_viz), str(loc))
                            self._phase = Phase.DROP_OBJECT

                            if possible_action == PossibleActions.DROPPING_A_BLOCK:
                                # for now the visualization of the dropped block is taken from the self.desired_objects list
                                # by getting the block that has a drop off location equal to the agent's current location
                                # update this after the memory of this agent is updated to be like the StrongAgent's memory
                                self._messageDroppedGoalBlock(str(obj_viz), str(loc))
                            else:
                                self._sendMessage(self.generateAMessageFromAction(possible_action), self.agent_name)

                # if not already dropped the object move to the next location
                if not flag:
                    self._state_tracker.update(state)

                    action = self._navigator.get_move_action(self._state_tracker)
                    # Move to the next location
                    if action != None:
                        return action, {}
                    else:
                        locations = list(map(lambda des_obj: des_obj[1], self.all_desired_objects))
                        self._navigator.reset_full()
                        # Add the navigation
                        self._navigator.add_waypoints(locations)
                        self.dropped_off_count = 0
                        self._phase = Phase.CHECK_ITEMS

            if Phase.DROP_OBJECT == self._phase:
                if self.object_to_be_dropped is None:
                    locations = list(map(lambda des_obj: des_obj[1], self.all_desired_objects))
                    self._navigator.reset_full()
                    # Add the navigation
                    self._navigator.add_waypoints(locations)
                    self.dropped_off_count = 0
                    self._phase = Phase.CHECK_ITEMS

                # update capacity
                else:
                    self.capacity -= 1
                    # Drop object
                    self._phase = Phase.FOLLOW_PATH_TO_DROP_OFF_LOCATION

                    return DropObject.__name__, {'object_id': self.object_to_be_dropped}

            if Phase.PLAN_PATH_TO_CLOSED_DOOR == self._phase:
                self._navigator.reset_full()

                doc = 0
                for key in self.at_drop_location:
                    doc += self.at_drop_location[key]
                if doc == len(self.all_desired_objects):
                    self._phase = Phase.GO_TO_REORDER_ITEMS
                    return None, {}

                if len(self.memory) > 0:
                    self._navigator.reset_full()
                    self._navigator.add_waypoints([self.memory[0]["location"]])
                    if len(self.not_dropped) > 0:
                        self.dropped_off_count = self.shortestDistance_drop(state, self.memory[0]["location"])
                    else:
                        self.dropped_off_count = -1
                    self.memory.pop(0)
                    self._phase = Phase.TRAVERSE_ROOM
                # Randomly pick a closed door or go to open room
                # Check if all rooms open
                else:
                    if len(self.rooms_to_visit) != 0:
                        self._door = self.rooms_to_visit.pop(random.randint(0, len(self.rooms_to_visit) - 1))[1]
                    elif len(self.visited) != 0:
                        self._door = self.visited.pop()[1]
                    else:
                        if len(self.all_rooms) == 0:
                            return None, {}

                        # get the first room, as they were sorted in the first iteration
                        room_name = self.all_rooms.pop(0)
                        self.all_rooms.append(room_name)
                        # get the door of the chosen room
                        self._door = [loc for loc in state.values()
                                      if "room_name" in loc and loc['room_name'] is
                                      room_name and 'class_inheritance' in loc and
                                      'Door' in loc['class_inheritance']][0]

                        if len(self._door) == 0:
                            return None, {}

                    # get the location of the door
                    doorLoc = self._door['location']

                    # Location in front of door is south from door
                    doorLoc = doorLoc[0], doorLoc[1] + 1

                    if len(self.not_dropped) > 0:
                        self.dropped_off_count = self.shortestDistance_drop(state, doorLoc)
                    else:
                        self.dropped_off_count = -1

                    # Send message of current action
                    possible_action = self.pickAnAction(PossibleActions.MOVING_TO_ROOM)
                    if possible_action == PossibleActions.MOVING_TO_ROOM:
                        self._messageMoveRoom(self._door['room_name'])
                    else:
                        self._sendMessage(self.generateAMessageFromAction(possible_action), self.agent_name)
                    # self._sendMessage('Moving to door of ' + self._door['room_name'], agent_name)
                    self._navigator.add_waypoints([doorLoc])
                    # go to the next phase
                    self._phase = Phase.FOLLOW_PATH_TO_CLOSED_DOOR

            if Phase.FOLLOW_PATH_TO_CLOSED_DOOR == self._phase:
                self._state_tracker.update(state)
                drop = self.check_for_not_dropped()
                if drop is not None:
                    return drop
                # Follow path to door
                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    return action, {}
                # go to the next phase
                self._phase = Phase.OPEN_DOOR

            if Phase.OPEN_DOOR == self._phase:
                self._phase = Phase.ENTER_ROOM

                # send a message to notify the others that you are opening a door
                possible_action = self.pickAnAction(PossibleActions.OPENING_DOOR)
                if possible_action == PossibleActions.OPENING_DOOR:
                    self._messageOpenDoor(self._door['room_name'])
                else:
                    self._sendMessage(self.generateAMessageFromAction(possible_action), self.agent_name)

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
                if len(self.all_desired_objects) != 0:
                    if state[self._state_tracker.agent_id]['location'] == self.all_desired_objects[0][1]:
                        self.all_desired_objects.pop(0)
                        self._phase = Phase.GRAB_AND_DROP

                    if self._phase != Phase.GRAB_AND_DROP:
                        self._state_tracker.update(state)
                        action = self._navigator.get_move_action(self._state_tracker)
                        # Move to the next location
                        if action != None:
                            return action, {}
                elif len(self.all_desired_objects) <= 0:
                    self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR

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

            if Phase.CHECK_ITEMS == self._phase:
                # Every time update the state for the new location of the agent
                self._state_tracker.update(state)

                action = self._navigator.get_move_action(self._state_tracker)

                for des, loc in self.all_desired_objects:
                    if state[self._state_tracker.agent_id]['location'] == loc:
                        for obj in state.get_closest_with_property("is_collectable"):
                            if obj["is_collectable"] is True and not 'GhostBlock' in obj['class_inheritance'] and obj[
                                "location"] == loc:
                                if self.compareObjects(des, obj):
                                    # self.dropped_off_count += 1
                                    self.at_drop_location[loc] = 1
                                    possible_action = self.pickAnAction(PossibleActions.ENCOUNTERING_A_BLOCK)
                                    if possible_action == PossibleActions.ENCOUNTERING_A_GOAL_BLOCK:
                                        self._messageFoundGoalBlock(str(des), str(loc))
                                    else:
                                        self._sendMessage(self.generateAMessageFromAction(possible_action),
                                                          self.agent_name)
                                    break
                        else:
                            continue
                        break
                if action is not None:
                    return action, {}
                else:
                    self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR

    def getLength(self):
        return 0.9

    def shortestDistance_drop(self, state, go_to):
        current_loc = state[self._state_tracker.agent_id]['location']
        if current_loc[0] > go_to[0]:
            x = current_loc[0] - go_to[0]
        else:
            x = go_to[0] - current_loc[0]

        if current_loc[1] > go_to[1]:
            y = current_loc[1] - go_to[1]
        else:
            y = go_to[1] - current_loc[1]

        distance = (x ** 2 + y ** 2) ** 0.5

        return int(round(distance * self.getLength()))

    def check_for_not_dropped(self):
        if self.dropped_off_count > 0:
            self.dropped_off_count -= 1
        elif self.dropped_off_count == 0:
            if len(self.not_dropped) > 0:
                if self.capacity > 0:
                    self.capacity -=1
                item = self.not_dropped.pop(0)[0]

                return DropObject.__name__, {'object_id': item}


    # pick an action from the PossibleActions with 20% chance it is the actual current action
    def pickAnAction(self, currentAction):
        if random.randint(1, 10) > 8:
            return currentAction
        return PossibleActions(random.randint(1, len(PossibleActions)))

    # generate a random message from the given action
    def generateAMessageFromAction(self, action):
        if action == PossibleActions.MOVING_TO_ROOM:
            return "Moving to " + (random.choice(self.all_rooms) if len(self.all_rooms) != 0 else "0?")
        elif action == PossibleActions.OPENING_DOOR:
            return "Opening door of " + (random.choice(self.all_rooms) if len(self.all_rooms) != 0 else "0?")
        elif action == PossibleActions.SEARCHING_A_ROOM:
            return "Searching through " + (random.choice(self.all_rooms) if len(self.all_rooms) != 0 else "0?")
        elif action == PossibleActions.ENCOUNTERING_A_GOAL_BLOCK:
            # when lying about finding, picking up, or dropping a goal block, use your own location, otherwise the location could be invalid
            return "Found goal block " + str(random.choice(self.all_desired_objects)[0]) + " at location " + str(
                self.state[self._state_tracker.agent_id]['location'])
        elif action == PossibleActions.PICKING_UP_A_BLOCK:
            return "Picking up goal block " + str(random.choice(self.all_desired_objects)[0]) + " at location " + str(
                self.state[self._state_tracker.agent_id]['location'])
        elif action == PossibleActions.DROPPING_A_BLOCK:
            return "Dropped goal block " + str(random.choice(self.all_desired_objects)[0]) + " at location " + str(
                self.state[self._state_tracker.agent_id]['location'])
        elif action == PossibleActions.ENCOUNTERING_A_BLOCK:
            # when lying about finding, picking up, or dropping a goal block, use your own location, otherwise the location could be invalid
            return "Found block " + str(random.choice((self.seenObjects + self.all_desired_objects))[0]) + " at location " + str(
                self.state[self._state_tracker.agent_id]['location'])
        else:
            print("Unexpected action received: ", action)
            exit(-1)

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
        self._sendMessage("Dropped goal block " + block_visualization + " at drop location " + location,
                          self.agent_name)

    def _messageFoundBlock(self, block_visualization, location):
        self._sendMessage("Found block " + block_visualization + " at location " + location, self.agent_name)


    def _init_trust_table(self, ids):
        data = {}
        for id in ids:
            arr = np.zeros(len(ids))
            arr.fill(0.5)
            data.update({id: arr})
        df = pd.DataFrame(data, index=ids, dtype=float)
        df.to_csv("Trust.csv")

    def _write_to_trust_table(self, trustor, trustee, new_trust):
        df = pd.read_csv('Trust.csv', index_col=0)
        df.loc[trustor, trustee] = new_trust
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
                    self.tbv.append((self.ticks, mssg.content, mssg.from_id))
                    self.acceptMessageIfSenderTrustworthy(mssg.content, mssg.from_id)
                    is_sequence_true = self.verify_action_sequence(self.receivedMessages, member, self.closed_doors)
                    if is_sequence_true is not None:
                        if is_sequence_true:
                            self.increaseTrust(member)
                        else:
                            self.decreaseTrust(member)
            self.already_said(mssg.content, mssg.from_id)
        tbv_copy = self.tbv
        for (ticks, mssg, from_id) in tbv_copy:
            is_true = self.checkMessageTrue(self.ticks, mssg, from_id)
            if is_true is not None:
                if is_true:
                    self.increaseTrust(from_id)
                else:
                    self.decreaseTrust(from_id)
                self.tbv.remove((ticks, mssg, from_id))

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

    def getObjectIdFromLocation(self, state, loc):
        for obj in state.get_closest_with_property("is_collectable"):
            if obj["is_collectable"] is True and \
                    not 'GhostBlock' in obj['class_inheritance'] and obj["location"] == loc:
                return obj["obj_id"]

    def acceptMessageIfSenderTrustworthy(self, mssg, sender):
        splitMssg = mssg.split(' ')
        if splitMssg[0] == 'Moving' and splitMssg[1] == 'to':
            room_to = splitMssg[2]
            if self.trustBeliefs[sender] >= 0.6:
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
            if self.trustBeliefs[sender] >= 0.6:
                for obj_vis, dropoff_loc in self.all_desired_objects:
                    if self.compareObjects(vis, obj_vis):
                        self.addToMemory(vis, loc, dropoff_loc)

        if splitMssg[0] == 'Dropped' and splitMssg[1] == 'goal':
            if self.trustBeliefs[sender] >= 0.6:
                pass

        if splitMssg[0] == 'Picking' and splitMssg[2] == 'goal':
            vis, loc = self.getVisLocFromMessage(mssg)
            if self.trustBeliefs[sender] >= 0.6:
                for dict1 in self.memory:
                    if self.compareObjects(dict1['visualization'], vis):
                        self.memory.remove(dict1)
                for obj in self.desired_objects:
                    if self.compareObjects(obj[0], vis):
                        self.desired_objects.remove(obj)

        if mssg.startswith("Trust score of "):
            agent, trust_score = self.getAgentScoreFromMessage(mssg)
            # sender will not send a trust score about itself, but check this is true just in case
            if trust_score is not None and self.trustBeliefs[sender] >= 0.6 and agent != self.agent_name and sender != agent: # TODO 0.6
                if self.trustBeliefs[agent] < trust_score - EPSILON:
                    self.trustBeliefs[agent] = np.clip(self.trustBeliefs[agent] + 0.05, 0, 1)
                    self._write_to_trust_table(self.agent_id, agent, self.trustBeliefs[agent])
                elif self.trustBeliefs[agent] > trust_score + EPSILON:
                    self.trustBeliefs[agent] = np.clip(self.trustBeliefs[agent] - 0.05, 0, 1)
                    self._write_to_trust_table(self.agent_id, agent, self.trustBeliefs[agent])
                # else don't do anything

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

    def getAgentScoreFromMessage(self, mssg):
        mssg_split = mssg.split()
        trust_score = None
        try:
            trust_score = float(mssg_split[5])
        except ValueError:
            print("ERROR when parsing message about trust. Incorrect message format: trust is not a float.")

        return mssg_split[3], trust_score

    def compareObjects(self, obj1, obj2):
        keys = ('shape', 'colour')
        flag = True
        for key in keys:
            if key in obj1 and key in obj2:
                if obj1[key] == obj2[key] or obj1[key] is None or obj2[key] is None:
                    pass
                else:
                    flag = False
        return flag

    def addToSeenObjects(self, obj):
        if obj not in self.seenObjects:
            self.seenObjects.append(obj)

    def shareTrustScores(self):
        for agent in self.trustBeliefs:
            if agent != self.agent_name:
                belief = self.trustBeliefs[agent]
                self._sendMessage("Trust score of " + agent + " is " + str(belief), self.agent_name)

    def verify_action_sequence(self, mssgs, sender, closed_doors):
        mssg, prev_mssg = self.find_mssg(mssgs, sender)

        if prev_mssg is not None:
            prev = prev_mssg.split(' ')
            # check if all door are open when a message for opening a door is received
            if (prev[0] == 'Opening' or mssg.split(' ')[0] == 'Opening') and len(closed_doors) == 0:
                return False

            # check moving to room, opening door sequence
            if prev[0] == 'Moving':
                curr = mssg.split(' ')

                # decrease trust score by little is action after moving to a room is not opening a door -> Lazy agent
                if curr[0] != 'Opening' and curr[2] not in closed_doors:
                    return False

                # decrease trust score if an agent says that he is going to one room, but opening the door of another
                if curr[0] == 'Opening' and prev[2] != curr[2]:
                    return False

            return True
        return

    def find_mssg(self, mssgs, from_id):
        counter = 0
        mssg = None
        prev_mssg = None
        for mssg in mssgs:
            # ignore messages about the trust score of an agent
            if mssg[2] == from_id and not "Trust score of " in mssg[1]:
                if (counter == 0):
                    mssg = mssg[1]
                    counter = counter + 1
                else:
                    prev_mssg = mssg[1]
                    break

        return mssg, prev_mssg

    def already_said(self, received_mssg, sender):
        for name in self.receivedMessages.keys():
            if name != self.agent_id and name != sender:
                for mssg in self.receivedMessages[name]:
                    vis, loc = self.getVisLocFromMessage(mssg[1])
                    received_vis, received_loc = self.getVisLocFromMessage(received_mssg[1])
                    if mssg[1].split(' ')[0] == 'Found':
                        if received_mssg[1].split(' ')[0] == 'Found':
                            if self.compareObjects(vis, received_vis) and loc == received_loc:
                                self.increaseTrust(sender)
                                self.increaseTrust(name)
                        elif received_mssg[1].split(' ')[0] == 'Picking':
                            if self.compareObjects(vis, received_vis) and loc == received_loc:
                                self.increaseTrust(sender)
                                self.increaseTrust(name)
                        elif received_mssg[1].split(' ')[0] == 'Dropped':
                            if self.compareObjects(vis, received_vis) and loc == received_loc:
                                self.increaseTrust(sender)
                                self.increaseTrust(name)

class ColorblindAgent(BW4TBrain):

    def __init__(self, settings: Dict[str, object]):
        super().__init__(settings)
        self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR
        self._teamMembers = []
        # This list only keeps track of the remaining desired_objects
        self.desired_objects = []
        # this list keeps track of all desired objects and is not updated throughout the execution of the simulation
        self.all_desired_objects = []
        self.agent_name = None
        # only the strong agents can pick 2 blocks
        # for other agents this is 0 or 1
        self.capacity = 0
        self.drop_off_locations = []
        self.object_to_be_dropped = None
        self.initialization_flag = True
        self.memory = []
        self.all_rooms = []

        self.ticks = 0
        self.receivedMessages = {}
        self.totalMessagesReceived = 0
        # A list of messages To Be Verified
        self.tbv = []
        # For each team member store trust score
        self.trustBeliefs = {}
        self.rooms_to_visit = []
        self.visited = []
        self.seenObjects = []
        self.dropped_off_count = 0
        self.at_drop_location = {}
        # used for the last phase - GRAB_AND_DROP to keep track of when an object is grabbed and after it was just dropped

        self.grab = False
        self.drop = False

        self.obj_id = None
        self.closed_doors = []

        self.not_dropped = []
        self.drop_counter = 0

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

        if self.ticks % 25 == 0:
            self.shareTrustScores()

        # We check if we enter for first time in the method as there is recursion
        # We want to keep track of some objects and reinitialize them every time
        if self.initialization_flag:
            agent_name = state[self.agent_id]['obj_id']
            self.agent_name = agent_name

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
                found_obj.append(({"shape": obj["visualization"]["shape"], "colour": None }, obj["location"]))
                self.at_drop_location[obj["location"]] = 0
            self.desired_objects = sorted(found_obj, key=lambda x: x[1], reverse=True)

            self.all_desired_objects = self.desired_objects.copy()
            sorted(self.all_desired_objects, key=lambda obj: obj[1], reverse=True)

            self.initTrustBeliefs()
            self._init_trust_table(state['World']['team_members'])

            for room in self.all_rooms:
                if room != "world_bounds":
                    door = state.get_room_doors(room)
                    self.rooms_to_visit.append((room, door[0]))

            self.closed_doors = [door for door in state.values()
                            if 'class_inheritance' in door and 'Door' in door['class_inheritance'] and not door[
                    'is_open']]
            for i, door in enumerate(self.closed_doors):
                self.closed_doors[i] = self.closed_doors[i]["room_name"]

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

                drop = self.check_for_not_dropped()
                if drop is not None:
                    return drop

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
                    found_obj.append(({"shape": obj["visualization"]["shape"], "colour": None }, obj["obj_id"], obj["location"]))

                # Check if some of the found objects that can be collected are desired objects
                check = False

                for obj in found_obj:
                    self.addToSeenObjects((obj[0], obj[2]))

                    for des, loc in self.desired_objects:
                        if obj[0]["shape"] == des["shape"]:
                            # In case they are desired objects for the strong agent we are interested only in the
                            # first two items from bottom to up, if they are we pick them
                            # in case they are not we save them in the memory for later use
                            self._messageFoundGoalBlock(str(obj[0]), str(obj[2]))
                            if (loc) in map(lambda o: o['location'], self.memory):
                                if self.capacity == 0:
                                    self.capacity += 1
                                    self.drop_off_locations.append((obj[0], obj[1], loc))
                                    self.desired_objects.remove((des, loc))
                                    self._messagePickUpGoalBlock(str(obj[0]), str(obj[2]))

                                    for dict1 in self.memory:
                                        if obj[0]["shape"] == dict1["visualization"]["shape"] and dict1["location"] == loc:
                                            self.memory.remove(dict1)

                                    return GrabObject.__name__, {'object_id': obj[1]}

                    if not check:
                        self._messageFoundBlock(str(obj[0]), str(obj[2]))
                        check = False

                # In case we are filled, deliver items, next phase
                if self.capacity == 1:
                    self._phase = Phase.DELIVER_ITEM
                # In case there is only one object left needed and is found deliver it, next phase
                # If no desired object was found just move

                if action != None:
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
                self._navigator.add_waypoints(locations)

                # Next phase
                self._phase = Phase.FOLLOW_PATH_TO_DROP_OFF_LOCATION

            # Follow path to the drop off location
            if Phase.FOLLOW_PATH_TO_DROP_OFF_LOCATION == self._phase:
                flag = False
                flag_not_dropped = False
                self.object_to_be_dropped = None
                # Check if the current location of the agent is the correct drop off location
                for obj_viz, obj_id, loc in self.drop_off_locations:
                    if state[self._state_tracker.agent_id]['location'] == loc:
                        flag = True
                        self.object_to_be_dropped = obj_id
                        # if it is the correct location drop the object
                        self._phase = Phase.DROP_OBJECT
                        self.drop_off_locations.remove((obj_viz, obj_id, loc))

                        for obj in state.get_closest_with_property("is_collectable"):
                            if obj["is_collectable"] is True and not 'GhostBlock' in obj['class_inheritance'] and obj[
                                "location"] == loc:
                                self.not_dropped.append((obj_id, loc))
                                flag_not_dropped = True
                                self.object_to_be_dropped = None

                        if not flag_not_dropped:
                            self.object_to_be_dropped = obj_id
                            self._messageDroppedGoalBlock(str(obj_viz), str(loc))
                            self._phase = Phase.DROP_OBJECT

                # if not already dropped the object move to the next location
                if not flag:
                    self._state_tracker.update(state)

                    action = self._navigator.get_move_action(self._state_tracker)
                    # Move to the next location
                    if action != None:
                        return action, {}
                    else:
                        locations = list(map(lambda des_obj: des_obj[1], self.all_desired_objects))
                        self._navigator.reset_full()
                        # Add the navigation
                        self._navigator.add_waypoints(locations)
                        self.dropped_off_count = 0
                        self._phase = Phase.CHECK_ITEMS

            if Phase.DROP_OBJECT == self._phase:
                if self.object_to_be_dropped is None:
                    locations = list(map(lambda des_obj: des_obj[1], self.all_desired_objects))
                    self._navigator.reset_full()
                    # Add the navigation
                    self._navigator.add_waypoints(locations)
                    self.dropped_off_count = 0
                    self._phase = Phase.CHECK_ITEMS
                # update capacity
                else:
                    self.capacity -= 1
                    # Drop object
                    self._phase = Phase.FOLLOW_PATH_TO_DROP_OFF_LOCATION

                    return DropObject.__name__, {'object_id': self.object_to_be_dropped}

            if Phase.PLAN_PATH_TO_CLOSED_DOOR == self._phase:
                self._navigator.reset_full()

                doc = 0
                for key in self.at_drop_location:
                    doc += self.at_drop_location[key]
                if doc == len(self.all_desired_objects):
                    self._phase = Phase.GO_TO_REORDER_ITEMS
                    return None, {}

                if len(self.memory) > 0:
                    self._navigator.reset_full()
                    self._navigator.add_waypoints([self.memory[0]["location"]])
                    if len(self.not_dropped) > 0:
                        self.dropped_off_count = self.shortestDistance_drop(state, self.memory[0]["location"])
                    else:
                        self.dropped_off_count = -1
                    self.memory.pop(0)
                    self._phase = Phase.TRAVERSE_ROOM
                # Randomly pick a closed door or go to open room
                # Check if all rooms open
                else:
                    if len(self.rooms_to_visit) != 0:
                        self._door = self.rooms_to_visit.pop(random.randint(0, len(self.rooms_to_visit) - 1))[1]
                    elif len(self.visited) != 0:
                        self._door = self.visited.pop()[1]
                    else:
                        if len(self.all_rooms) == 0:
                            return None, {}

                        # get the first room, as they were sorted in the first iteration
                        room_name = self.all_rooms.pop(0)
                        self.all_rooms.append(room_name)
                        # get the door of the chosen room
                        self._door = [loc for loc in state.values()
                                      if "room_name" in loc and loc['room_name'] is
                                      room_name and 'class_inheritance' in loc and
                                      'Door' in loc['class_inheritance']]

                        if len(self._door) == 0:
                            return None, {}

                        self._door = self._door[0]

                    doorLoc = self._door['location']

                    # Location in front of door is south from door
                    doorLoc = doorLoc[0], doorLoc[1] + 1

                    if len(self.not_dropped) > 0:
                        self.dropped_off_count = self.shortestDistance_drop(state, doorLoc)
                    else:
                        self.dropped_off_count = -1

                    # Send message of current action
                    self._messageMoveRoom(self._door['room_name'])
                    # self._sendMessage('Moving to door of ' + self._door['room_name'], agent_name)
                    self._navigator.add_waypoints([doorLoc])
                    # go to the next phase
                    self._phase = Phase.FOLLOW_PATH_TO_CLOSED_DOOR

            if Phase.FOLLOW_PATH_TO_CLOSED_DOOR == self._phase:
                self._state_tracker.update(state)
                drop = self.check_for_not_dropped()
                if drop is not None:
                    return drop
                # Follow path to door
                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    return action, {}
                # go to the next phase
                self._phase = Phase.OPEN_DOOR

            if Phase.OPEN_DOOR == self._phase:
                self._phase = Phase.ENTER_ROOM

                # send a message to notify the others that you are opening a door
                self._messageOpenDoor(self._door['room_name'])

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
                if len(self.all_desired_objects) != 0:
                    if state[self._state_tracker.agent_id]['location'] == self.all_desired_objects[0][1]:
                        self.all_desired_objects.pop(0)
                        self._phase = Phase.GRAB_AND_DROP

                    if self._phase != Phase.GRAB_AND_DROP:
                        self._state_tracker.update(state)
                        action = self._navigator.get_move_action(self._state_tracker)
                        # Move to the next location
                        if action != None:
                            return action, {}
                elif len(self.all_desired_objects) <= 0:
                    self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR

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

            if Phase.CHECK_ITEMS == self._phase:
                # Every time update the state for the new location of the agent
                self._state_tracker.update(state)

                action = self._navigator.get_move_action(self._state_tracker)

                for des, loc in self.all_desired_objects:
                    if state[self._state_tracker.agent_id]['location'] == loc:
                        for obj in state.get_closest_with_property("is_collectable"):
                            if obj["is_collectable"] is True and not 'GhostBlock' in obj['class_inheritance'] and obj[
                                "location"] == loc:
                                if self.compareObjects(des, obj):
                                    self.at_drop_location[loc] = 1
                                    break
                        else:
                            continue
                        break
                if action is not None:
                    return action, {}
                else:
                    self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR

    def getRandom1(self):
        return 0.9

    def getLength(self):
        return 0.9

    def shortestDistance_drop(self, state, go_to):
        current_loc = state[self._state_tracker.agent_id]['location']
        if current_loc[0] > go_to[0]:
            x = current_loc[0] - go_to[0]
        else:
            x = go_to[0] - current_loc[0]

        if current_loc[1] > go_to[1]:
            y = current_loc[1] - go_to[1]
        else:
            y = go_to[1] - current_loc[1]

        distance = (x ** 2 + y ** 2) ** 0.5

        return int(round(distance * self.getLength()))

    def check_for_not_dropped(self):
        if self.dropped_off_count > 0:
            self.dropped_off_count -= 1
        elif self.dropped_off_count == 0:
            if len(self.not_dropped) > 0:
                if self.capacity > 0:
                    self.capacity -= 1
                item = self.not_dropped.pop(0)[0]

                return DropObject.__name__, {'object_id': item}


    def addToMemory_color(self, vis, loc, drop):
        if len(self.memory) == 0:
            self.memory.append({"visualization": vis,
                                "location": loc,
                                "drop_off_location": drop})
        flag_check = True
        for v in self.memory:
            if v["visualization"]["shape"] == vis["shape"] and v["drop_off_location"] == drop:
                flag_check = False

        if flag_check:
            self.memory.append({"visualization": vis,
                                "location": loc,
                                "drop_off_location": drop})

        self.memory = sorted(self.memory, key=lambda x: x["drop_off_location"],
                             reverse=True)

    def getObjectIdFromLocation(self, state, loc):
        for obj in state.get_closest_with_property("is_collectable"):
            if obj["is_collectable"] is True and not 'GhostBlock' in obj['class_inheritance'] and obj[
                "location"] == loc:
                return obj["obj_id"]
        return

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
        self._sendMessage("Dropped goal block " + block_visualization + " at drop location " + location,
                          self.agent_name)

    def _messageFoundBlock(self, block_visualization, location):
        self._sendMessage("Found block " + block_visualization + " at location " + location, self.agent_name)

    def _init_trust_table(self, ids):
        data = {}
        for id in ids:
            arr = np.zeros(len(ids))
            arr.fill(0.5)
            data.update({id: arr})
        df = pd.DataFrame(data, index=ids, dtype=float)
        df.to_csv("Trust.csv")

    def _write_to_trust_table(self, trustor, trustee, new_trust):
        df = pd.read_csv('Trust.csv', index_col=0)
        df.loc[trustor, trustee] = new_trust
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
                    if (self.ticks, mssg.content, mssg.from_id) not in self.tbv and mssg.content.startswith('Found'):
                        self.tbv.append((self.ticks, mssg.content, mssg.from_id))
                    self.acceptMessageIfSenderTrustworthy(mssg.content, mssg.from_id)
                    is_sequence_true = self.verify_action_sequence(self.receivedMessages, member)
                    if is_sequence_true is not None:
                        if is_sequence_true:
                            self.increaseTrust(member)
                        else:
                            self.decreaseTrust(member)
            self.already_said(mssg.content, mssg.from_id)
        tbv_copy = self.tbv
        for (ticks, mssg, from_id) in tbv_copy:
            if mssg.startswith('Found'):
                is_true = self.checkMessageTrue(self.ticks, mssg, from_id)
                if is_true is not None:
                    if is_true:
                        self.increaseTrust(from_id)
                    else:
                        self.decreaseTrust(from_id)
                    self.tbv.remove((ticks, mssg, from_id))

    def initTrustBeliefs(self):
        for member in self._teamMembers:
            self.trustBeliefs[member] = 0.5


    def acceptMessageIfSenderTrustworthy(self, mssg, sender):
        splitMssg = mssg.split(' ')
        if splitMssg[0] == 'Moving' and splitMssg[1] == 'to':
            room_to = splitMssg[2]
            if self.trustBeliefs[sender] >= 0.6:
                for room, door in self.rooms_to_visit:
                    if room_to == room:
                        self.rooms_to_visit.remove((room, door))
                        self.visited.append((room, door))
                        # remove door of room from all closed_doors
                        self.closed_doors.remove(door["room_name"])

        if splitMssg[0] == 'Opening' and splitMssg[1] == 'door':
            pass

        if splitMssg[0] == 'Searching' and splitMssg[1] == 'through':
            pass

        if splitMssg[0] == 'Found' and splitMssg[1] == 'goal':
            vis, loc = self.getVisLocFromMessage(mssg)
            if self.trustBeliefs[sender] >= 0.6:
                for obj_vis, dropoff_loc in self.all_desired_objects:
                    if self.compareObjects(vis, obj_vis):
                        self.addToMemory_color(vis, loc, dropoff_loc)

        if splitMssg[0] == 'Dropped' and splitMssg[1] == 'goal':
            if self.trustBeliefs[sender] >= 0.6:
                pass

        if splitMssg[0] == 'Picking' and splitMssg[2] == 'goal':
            vis, loc = self.getVisLocFromMessage(mssg)
            if self.trustBeliefs[sender] >= 0.6:
                for dict1 in self.memory:
                    if self.compareObjects(dict1['visualization'], vis):
                        self.memory.remove(dict1)
                for obj in self.desired_objects:
                    if self.compareObjects(obj[0], vis):
                        self.desired_objects.remove(obj)

        if mssg.startswith("Trust score of "):
            agent, trust_score = self.getAgentScoreFromMessage(mssg)
            # sender will not send a trust score about itself, but check this is true just in case
            if trust_score is not None and self.trustBeliefs[sender] >= 0.6 and agent != self.agent_name and sender != agent: # TODO 0.6
                if self.trustBeliefs[agent] < trust_score - EPSILON:
                    self.trustBeliefs[agent] = np.clip(self.trustBeliefs[agent] + 0.05, 0, 1)
                    self._write_to_trust_table(self.agent_id, agent, self.trustBeliefs[agent])
                elif self.trustBeliefs[agent] > trust_score + EPSILON:
                    self.trustBeliefs[agent] = np.clip(self.trustBeliefs[agent] - 0.05, 0, 1)
                    self._write_to_trust_table(self.agent_id, agent, self.trustBeliefs[agent])
                # else don't do anything

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

    def getAgentScoreFromMessage(self, mssg):
        mssg_split = mssg.split()
        trust_score = None
        try:
            trust_score = float(mssg_split[5])
        except ValueError:
            print("ERROR when parsing message about trust. Incorrect message format: trust is not a float.")

        return mssg_split[3], trust_score

    def compareObjects(self, obj1, obj2):
        keys = ('shape', 'colour')
        flag = True
        for key in keys:
            if key in obj1 and key in obj2:
                if obj1[key] == obj2[key] or obj1[key] is None or obj2[key] is None:
                    pass
                else:
                    flag = False
        return flag

    def addToSeenObjects(self, obj):
        if obj not in self.seenObjects:
            self.seenObjects.append(obj)

    def verify_action_sequence(self, mssgs, sender):
        mssg, prev_mssg = self.find_mssg(mssgs, sender)

        if prev_mssg is not None:
            prev = prev_mssg.split(' ')
            curr = mssg.split(' ')
            # check if all door are open when a message for opening a door is received
            if (prev[0] == 'Opening' or curr[0] == 'Opening') and len(self.closed_doors) == 0:
                return False

            if (prev[0] == 'Opening' and prev[3] not in self.closed_doors) or (
                    curr[0] == 'Opening' and curr[3] not in self.closed_doors):
                return False

            # check moving to room, opening door sequence
            if prev[0] == 'Moving':
                # decrease trust score by little is action after moving to a room is not opening a door -> Lazy agent
                if curr[0] != 'Opening':
                    return False

                # decrease trust score if an agent says that he is going to one room, but opening the door of another
                if curr[0] == 'Opening' and prev[2] != curr[3]:
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

    def already_said(self, received_mssg, sender):
        for name in self.receivedMessages.keys():
            if name != self.agent_id and name != sender:
                for mssg in self.receivedMessages[name]:
                    vis, loc = self.getVisLocFromMessage(mssg[1])
                    received_vis, received_loc = self.getVisLocFromMessage(received_mssg[1])
                    if mssg[1].split(' ')[0] == 'Found':
                        if received_mssg[1].split(' ')[0] == 'Found':
                            if self.compareObjects(vis, received_vis) and loc == received_loc:
                                self.increaseTrust(sender)
                                self.increaseTrust(name)
                        elif received_mssg[1].split(' ')[0] == 'Picking':
                            if self.compareObjects(vis, received_vis) and loc == received_loc:
                                self.increaseTrust(sender)
                                self.increaseTrust(name)
                        elif received_mssg[1].split(' ')[0] == 'Dropped':
                            if self.compareObjects(vis, received_vis) and loc == received_loc:
                                self.increaseTrust(sender)
                                self.increaseTrust(name)

    def shareTrustScores(self):
        for agent in self.trustBeliefs:
            if agent != self.agent_name:
                belief = self.trustBeliefs[agent]
                self._sendMessage("Trust score of " + agent + " is " + str(belief), self.agent_name)

