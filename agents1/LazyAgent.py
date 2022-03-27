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
    TRAVERSE_ROOM = 5,
    DELIVER_ITEM = 6,
    FOLLOW_PATH_TO_DROP_OFF_LOCATION = 7,
    DROP_OBJECT = 8


class LazyAgent(BW4TBrain):

    def __init__(self, settings: Dict[str, object]):
        super().__init__(settings)
        self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR
        self._teamMembers = []
        self.desired_objects = []
        self.agent_name = None
        # only the strong agents can pick 2 blocks
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

    def initialize(self):
        super().initialize()
        self._state_tracker = StateTracker(agent_id=self.agent_id)
        self._navigator = Navigator(agent_id=self.agent_id,
                                    action_set=self.action_set, algorithm=Navigator.A_STAR_ALGORITHM)

    def filter_bw4t_observations(self, state):
        return state

    def decide_on_bw4t_action(self, state: State):
        agent_name = state[self.agent_id]['obj_id']
        self.agent_name = agent_name
        # Add team members
        for member in state['World']['team_members']:
            if member != agent_name and member not in self._teamMembers:
                self._teamMembers.append(member)
                # Process messages from team members
        receivedMessages = self._processMessages(self._teamMembers)
        # Update trust beliefs for team members
        self._trustBelief(self._teamMembers, receivedMessages)

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
                found_obj.append((obj["visualization"], obj["location"]))
            self.desired_objects = sorted(found_obj, key=lambda x: x[1], reverse=True)

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
                    for obj in found_obj:
                        for des, loc in self.desired_objects:
                            if obj[0]["shape"] == des["shape"] and obj[0]["colour"] == des["colour"]:
                                # In case they are desired objects for the strong agent we are interested only in the
                                # first two items from bottom to up, if they are we pick them
                                # in case they are not we save them in the memory for later use
                                self._messageFoundGoalBlock(str(obj[0]), str(loc))
                                # print(self.desired_objects[0])
                                # print((des, loc))

                                if ((des, loc)) == self.desired_objects[0]:
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

                                            for num, dict1 in enumerate(self.memory):
                                                if obj[0]["shape"] == dict1["visualization"]["shape"] \
                                                        and obj[0]["colour"] == dict1["visualization"]["colour"]:
                                                    self.memory.remove(dict1)

                                            return GrabObject.__name__, {'object_id': obj[1]}

                                        self.addToMemory(obj[0], obj[2], loc)
                                        self.memory = sorted(self.memory, key=lambda x: x["drop_off_location"],
                                                             reverse=True)

                                elif ((des, loc)) in self.desired_objects:
                                    self.addToMemory(obj[0], obj[2], loc)
                                    #
                                    # self.memory.append({"visualization": obj[0],
                                    #                         "location": obj[2],
                                    #                         "drop_off_location": loc})
                                    self.memory = sorted(self.memory, key=lambda x: x["drop_off_location"],
                                                                 reverse=True)
                                    print("MEMORY", self.memory)



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
                # Check if the current location of the agent is the correct drop off location
                # print(self.drop_off_locations)

                if self.stop_when == 0:
                    self._phase = Phase.DROP_OBJECT
                    self._messageDroppedGoalBlock(str(self.my_object[0]),
                                                  str(state[self._state_tracker.agent_id]['location']))
                    self.addToMemory(self.my_object[0], state[self._state_tracker.agent_id]['location'], self.my_object[1])

                    # self.memory.append({"visualization": self.my_object[0],
                    #                     "location": state[self._state_tracker.agent_id]['location'],
                    #                     "drop_off_location": self.my_object[1]})

                    self.memory = sorted(self.memory, key=lambda x: x["drop_off_location"], reverse=True)
                    self.object_to_be_dropped = self.my_object_id

                elif state[self._state_tracker.agent_id]['location'] == self.drop_off_locations[2]:
                    flag = True

                    self.object_to_be_dropped = self.drop_off_locations[1]

                    # if it is the correct location drop the object

                    self._phase = Phase.DROP_OBJECT

                    self.desired_objects.pop(0)

                    self._messageDroppedGoalBlock(str(self.drop_off_locations[0]), str(self.drop_off_locations[2]))

                    # for num, obj in enumerate(self.desired_objects):
                    #     if self.drop_off_locations[0]["shape"] == obj[0]["shape"] \
                    #             and self.drop_off_locations[0]["colour"] == obj[0]["colour"]:
                    #         self.memory.remove(obj)

                    print("MEMORY", self.memory)

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
                # print("dropped object")
                # Drop object

                self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR

                return DropObject.__name__, {'object_id': self.object_to_be_dropped}

            if Phase.PLAN_PATH_TO_CLOSED_DOOR == self._phase:
                decision = self.getRandom50()
                print("USE MEMORY", self.use_memory)
                if len(self.memory) > 0 and self.desired_objects[0][0]["shape"] == \
                        self.memory[0]["visualization"]["shape"] \
                        and self.desired_objects[0][0]["colour"] == self.memory[0]["visualization"]["colour"]\
                        and self.use_memory:

                    print("GO TO MEMORY: ", decision)
                    print("MEMORY", self.memory)
                    self._navigator.reset_full()
                    self._navigator.add_waypoints([self.memory[0]["location"]])

                    self._phase = Phase.TRAVERSE_ROOM
                    if decision:
                        self.stop_when = -1
                        self.use_memory = True
                        print("GO TO OBJECT", self.memory[0])
                    else:
                        distance = self.getRandom1() * self.shortestDistance(
                            state[self._state_tracker.agent_id]['location'], self.memory[0]["location"])

                        self.stop_when = int(round(distance * self.getRandom1()))
                        print("STOP when", self.stop_when)
                        self.use_memory = False
                else:
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
                        self.all_rooms.append(room_name)
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

                    self.use_memory = True
                    # Send message of current action
                    decision = self.getRandom50()
                    print("GO TO ROOM: ", decision)

                    self._navigator.add_waypoints([doorLoc])
                    if decision:
                        self._messageMoveRoom(self._door['room_name'])
                        self._sendMessage('Moving to door of ' + self._door['room_name'], agent_name)
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

    def getRandom50(self):
        return random.random() > 0.5

    def getRandom1(self):
        return random.random()

    def getRandomAction(self):
        return random.randint(1, 8)

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