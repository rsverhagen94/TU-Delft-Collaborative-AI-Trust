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


class PossibleActions(enum.Enum):
    MOVING_TO_ROOM = 1
    OPENING_DOOR = 2
    SEARCHING_A_ROOM = 3
    ENCOUNTERING_A_BLOCK = 4
    PICKING_UP_A_BLOCK = 5
    DROPPING_A_BLOCK = 6


class LiarAgent(BW4TBrain):

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
        self.memory = None
        self.all_rooms = []

        # keep track of already encountered items to lie about which was seen where
        self.already_encountered_items = []

        self._previous_phase = None
        self.grid_shape = None

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
                found_obj.append((obj["visualization"], obj["location"]))
            self.desired_objects = sorted(found_obj, key=lambda x: x[1], reverse=True)

            self.grid_shape = state["grid_shape"]["grid_shape"]
            print("GRID", PossibleActions(3))

        # TODO send messages for all cases
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
                self._navigator.add_waypoints(sorted_by_xy)

                # Go to the next phase
                self._phase = Phase.TRAVERSE_ROOM

            if Phase.TRAVERSE_ROOM == self._phase:
                if self._previous_phase != self._phase:
                    possible_action = self.pickAnAction(PossibleActions.SEARCHING_A_ROOM)
                    if possible_action  == PossibleActions.SEARCHING_A_ROOM:
                        # be honest
                        self._sendMessage("Searching through " + room)
                    else:
                        self._sendMessage(self.generateAMessageFromAction(possible_action ))

                self._previous_phase = self._phase
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
                            if obj[0]["shape"] == des["shape"] and obj[0]["colour"] == des["colour"]:
                                # In case they are desired objects for the strong agent we are interested only in the
                                # first two items from bottom to up, if they are we pick them
                                # in case they are not we save them in the memory for later use
                                if ((des, loc)) != self.desired_objects[0] \
                                        and ((des, loc)) in self.desired_objects:
                                    # Note a small bug was found. It does not find and pick object
                                    # when the memory is pointing to the middle room (room 5).
                                    # In all other cases it work properly
                                    self.memory = state[self._state_tracker.agent_id]['location']

                                    possible_action = self.pickAnAction(PossibleActions.ENCOUNTERING_A_BLOCK)
                                    if possible_action == PossibleActions.ENCOUNTERING_A_BLOCK:
                                        self._sendMessage("Found goal block " + str(obj[0]) + " at location " + str(obj[2]))
                                    else:
                                        self._sendMessage(self.generateAMessageFromAction(possible_action))
                                else:
                                    # Grab object if there is a capacity
                                    if self.capacity < 1:
                                        self.capacity += 1
                                        self.drop_off_locations.append((obj[1], loc))
                                        self.desired_objects.remove((des, loc))

                                        possible_action  = self.pickAnAction(PossibleActions.PICKING_UP_A_BLOCK)
                                        if possible_action  == PossibleActions.PICKING_UP_A_BLOCK:
                                            # be honest
                                            self._sendMessage("Picking up goal block " + str(des) + " at location " + str(loc))
                                        else:
                                            self._sendMessage(self.generateAMessageFromAction(possible_action))

                                        return GrabObject.__name__, {'object_id': obj[1]}

                    # In case we are filled, deliver items, next pahse
                    if self.capacity > 0:
                        self._phase = Phase.DELIVER_ITEM
                    # In case there is only one object left needed and is found deliver it, next phase
                    # elif len(self.desired_objects) < 2 and self.capacity > 0:
                    #     self._phase = Phase.DELIVER_ITEM

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
                        self.object_to_be_dropped = (obj_id, loc)
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
                        if self.memory is not None:
                            self._navigator.reset_full()
                            self._navigator.add_waypoints([self.memory])
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

                possible_action  = self.pickAnAction(PossibleActions.DROPPING_A_BLOCK)
                if possible_action  == PossibleActions.DROPPING_A_BLOCK:
                    # TODO cheating a little bit here, since the visualization is not saved anywhere
                    # for now the visualization of the dropped block is taken from the self.desired_objects list
                    # by getting the block that has a drop off location equal to the agent's current location
                    # update this after the memory of this agent is updated to be like the StrongAgent's memory
                    agent_location = state[self._state_tracker.agent_id]['location']
                    self._sendMessage("Dropped goal block " + str(list(filter(lambda block: block[1] == agent_location, self.desired_objects))[0]) + " at location " + str(agent_location))
                else:
                    self._sendMessage(self.generateAMessageFromAction(possible_action))

                return DropObject.__name__, {'object_id': self.object_to_be_dropped[0]}

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
                possible_action = self.pickAnAction(PossibleActions.MOVING_TO_ROOM)
                if possible_action == PossibleActions.MOVING_TO_ROOM:
                    self._sendMessage("Moving to " + self._door['obj_id'])
                else:
                    self._sendMessage(self.generateAMessageFromAction(possible_action))

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

                # send a message to notify the others that you are opening a door
                possible_action = self.pickAnAction(PossibleActions.OPENING_DOOR)
                if possible_action == PossibleActions.OPENING_DOOR:
                    self._sendMessage("Opening door of " + self._door['obj_id'])
                else:
                    self._sendMessage(self.generateAMessageFromAction(possible_action))

                # Open door
                # If already opened, no change
                return OpenDoorAction.__name__, {'object_id': self._door['obj_id']}

    # pick an action from the PossibleActions with 20% chance it is the actual current action
    def pickAnAction(self, currentAction):
        if random.randint(1,10) > 8:
            return currentAction
        return PossibleActions(random.randint(1, len(PossibleActions)))

    # generate a random message from the given action
    def generateAMessageFromAction(self, action):
        print("generating a message from action")
        if action == PossibleActions.MOVING_TO_ROOM:
            return "Moving to " + (random.choice(self.all_rooms) if len(self.all_rooms) != 0 else "0?")
        elif action == PossibleActions.OPENING_DOOR:
            return "Opening door of " + (random.choice(self.all_rooms) if len(self.all_rooms) != 0 else "0?")
        elif action == PossibleActions.SEARCHING_A_ROOM:
            return "Searching through " + (random.choice(self.all_rooms) if len(self.all_rooms) != 0 else "0?")
        elif action == PossibleActions.ENCOUNTERING_A_BLOCK:
            # when lying about finding, picking up, or dropping a goal block, use your own location, otherwise the location could be invalid
            return "Found goal block " + str(random.choice(self.desired_objects)[0]) + " at location " + str(self.state[self._state_tracker.agent_id]['location'])
        elif action == PossibleActions.PICKING_UP_A_BLOCK:
            return "Picking up goal block " + str(random.choice(self.desired_objects)[0]) + " at location " + str(self.state[self._state_tracker.agent_id]['location'])
        elif action == PossibleActions.DROPPING_A_BLOCK:
            return "Dropped goal block " + str(random.choice(self.desired_objects)[0]) + " at location " + str(self.state[self._state_tracker.agent_id]['location'])
        else:
            print("Unexpected action received: ", action)
            exit(-1)
        return "Unreachable"

    def _sendMessage(self, mssg):
        '''
        Enable sending messages in one line of code
        '''
        msg = Message(content=mssg, from_id=self.agent_id)
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
        for x in range(min_xy[0] + 1, max_xy[0]):
            for y in range(min_xy[1] + 1, max_xy[1] - 1):
                list_coordinates.append((x, y))
                # print(x, y)

        self._navigator.add_waypoints(list_coordinates)
