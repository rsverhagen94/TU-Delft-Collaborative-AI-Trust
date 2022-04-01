from typing import final, List, Dict, Final
import enum, random
from bw4t.BW4TBrain import BW4TBrain
from matrx.agents.agent_utils.state import State
from matrx.agents.agent_utils.navigator import Navigator
from matrx.agents.agent_utils.state_tracker import StateTracker
from matrx.actions.door_actions import OpenDoorAction
from matrx.actions.object_actions import GrabObject, DropObject
from matrx.messages.message import Message
from matrx.actions.action import Action


class Phase(enum.Enum):
    PLAN_PATH_TO_ROOM = 1,
    FOLLOW_PATH_TO_ROOM = 2,
    OPEN_DOOR = 3,
    WAIT_FOR_DOOR = 4.
    ENTERING_ROOM = 5,
    SEARCHING_ROOM = 6,
    FOLLOW_PATH_TO_DROP = 7,
    DROP_OBJECT = 8,
    CHECK_GOALS = 9,
    FOLLOW_PATH_TO_GOAL = 10,
    PICK_UP_GOAL_BLOCK = 11,
    PUT_AWAY_WRONG_BLOCK = 12,
    WAIT_FOR_FINISH = 13,
    MOVE_GOAL_BLOCK = 14,
    UPDATE_GOAL_LIST = 15,
    MOVING_TO_KNOWN_BLOCK = 16,
    RUN_BACK = 17


class StrongAgent(BW4TBrain):

    def __init__(self, settings: Dict[str, object]):
        super().__init__(settings)
        self._phase = Phase.PLAN_PATH_TO_ROOM
        self._teamMembers = []

    def initialize(self):
        super().initialize()
        self._state_tracker = StateTracker(agent_id=self.agent_id)
        self._navigator = Navigator(agent_id=self.agent_id,
                                    action_set=self.action_set, algorithm=Navigator.A_STAR_ALGORITHM)
        self._goalBlocks = None
        self._goalsInitialized = False
        self._carrying = None
        self._carryingO = None
        self._goalsWrong = []
        self._checkGoals = []
        self._possibleGoalBLocks = []
        self._notExplored = []

    def filter_bw4t_observations(self, state):
        return state

    def decide_on_bw4t_action(self, state: State):
        if not self._goalsInitialized:
            self._goalBlocks = state.get_with_property({'is_goal_block': True})
            self._goalsInitialized = True
            self._notExplored = [door for door in state.values()
                               if 'class_inheritance' in door and 'Door' in door['class_inheritance']]

        agent_name = state[self.agent_id]['obj_id']
        # Add team members
        for member in state['World']['team_members']:
            if member != agent_name and member not in self._teamMembers:
                self._teamMembers.append(member)
                # Process messages from team members
        receivedMessages = self._processMessages(self._teamMembers)
        # Update trust beliefs for team members
        self._trustBlief(self._teamMembers, receivedMessages)

        objects = state.get_closest_with_property({'class_inheritance': ['CollectableBlock']})
        if objects is not None:
            for o in objects:
                count = 0
                for g in self._goalBlocks:
                    if o['location'] == g['location']:
                        count += 1
                if count == 0:
                    for g in self._goalBlocks:
                        if o['visualization']['shape'] == g['visualization']['shape'] and o['visualization']['colour'] == \
                                g['visualization']['colour'] and len(o['carried_by']) == 0:
                            self._possibleGoalBLocks.append(o)
                            self._sendMessage('Found goal block {\"size\": ' + str(
                                o['visualization']['size']) + ', \"shape\": ' + str(
                                o['visualization']['shape']) + ', \"colour\": ' + str(
                                o['visualization']['colour']) + '} at location ' + str(o['location']), agent_name)

        while True:
            if Phase.PLAN_PATH_TO_ROOM == self._phase:
                self._navigator.reset_full()
                rooms = [door for door in state.values()
                               if 'class_inheritance' in door and 'Door' in door['class_inheritance']]
                if len(rooms) == 0:
                    return None, {}
                # If not every room has been explored, pick one randomly from the non explored, else pick randomly pick a door
                if len(self._notExplored) != 0:
                    self._door = random.choice(self._notExplored)
                else:
                    self._door = random.choice(rooms)
                doorLoc = self._door['location']
                # Location in front of door is south from door
                doorLoc = doorLoc[0], doorLoc[1] + 1
                # Send message of current action
                self._sendMessage('Moving to ' + self._door['room_name'], agent_name)
                self._navigator.add_waypoints([doorLoc])
                self._phase = Phase.FOLLOW_PATH_TO_ROOM

            if Phase.FOLLOW_PATH_TO_ROOM == self._phase:
                self._state_tracker.update(state)
                # Follow path to room
                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    return action, {}
                if self._door['is_open']:
                    self._phase = Phase.ENTERING_ROOM
                else:
                     self._phase = Phase.OPEN_DOOR

            if Phase.OPEN_DOOR == self._phase:
                self._phase = Phase.WAIT_FOR_DOOR
                # Open door
                self._sendMessage('Opening door of ' + self._door['room_name'], agent_name)
                return OpenDoorAction.__name__, {'object_id': self._door['obj_id']}

            if Phase.WAIT_FOR_DOOR == self._phase:
                self._phase = Phase.ENTERING_ROOM
                return None, {}

            if Phase.ENTERING_ROOM == self._phase:
                if self._door in self._notExplored:
                    self._notExplored.remove(self._door)
                self._sendMessage('Searching through ' + self._door['room_name'], agent_name)
                self._navigator.reset_full()
                objects = state.get_room_objects(self._door['room_name'])
                for o in objects:
                    if o['is_traversable']:
                        self._navigator.add_waypoints([o['location']])
                self._phase = Phase.SEARCHING_ROOM

            if Phase.SEARCHING_ROOM == self._phase:
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                objects = state.get_closest_with_property({'class_inheritance': ['CollectableBlock']})
                if objects is not None:
                    for o in objects:
                        for g in self._goalBlocks:
                            if o['visualization']['shape'] == g['visualization']['shape'] and o['visualization']['colour'] == g['visualization']['colour'] and len(o['carried_by']) == 0:
                                self._sendMessage('Found goal block {\"size\": ' + str(
                                    o['visualization']['size']) + ', \"shape\": ' + str(
                                    o['visualization']['shape']) + ', \"colour\": ' + str(
                                    o['visualization']['colour']) + '} at location ' + str(o['location']), agent_name)
                                self._sendMessage('Picking up goal block {\"size\": ' + str(
                                    o['visualization']['size']) + ', \"shape\": ' + str(
                                    o['visualization']['shape']) + ', \"colour\": ' + str(
                                    o['visualization']['colour']) + '} at location ' + str(
                                    o['location']), agent_name)
                                self._phase = Phase.FOLLOW_PATH_TO_DROP
                                self._navigator.reset_full()
                                self._navigator.add_waypoints([g['location']])
                                action = GrabObject.__name__
                                action_kwargs = {}
                                action_kwargs['object_id'] = o['obj_id']
                                self._carrying = g
                                self._carryingO = o
                                return action, action_kwargs
                if action!=None:
                    return action, {}
                if len(self._possibleGoalBLocks) == 0:
                    self._phase = Phase.PLAN_PATH_TO_ROOM
                else:
                    block = self._possibleGoalBLocks[0]
                    self._navigator.reset_full()
                    self._navigator.add_waypoints([block['location']])
                    self._phase = Phase.MOVING_TO_KNOWN_BLOCK

            if Phase.FOLLOW_PATH_TO_DROP == self._phase:
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    return action, {}
                self._phase = Phase.DROP_OBJECT

            if Phase.DROP_OBJECT == self._phase:
                if len(self._possibleGoalBLocks) == 0:
                    self._phase = Phase.PLAN_PATH_TO_ROOM
                else:
                    block = self._possibleGoalBLocks[0]
                    self._navigator.reset_full()
                    self._navigator.add_waypoints([block['location']])
                    self._phase = Phase.MOVING_TO_KNOWN_BLOCK

                # If there already is a block in this location, move to a different location and drop the block there.
                objects = state.get_closest_with_property({'class_inheritance': ['CollectableBlock']})
                if objects is not None:
                    for o in objects:
                        if o['location'] == self.state.get_self()['location']:
                            self._navigator.reset_full()
                            self._navigator.add_waypoints([self._carryingO['location']])
                            self._phase = Phase.FOLLOW_PATH_TO_DROP
                            return None, {}

                self._goalBlocks.remove(self._carrying)
                self._sendMessage('Removing block' + str(self._carrying) + ' from list: ' + str(self._goalBlocks),
                                  agent_name)

                if len(self._goalBlocks) >= 1:
                    self._sendMessage('Updating goal list with ' + str(len(self._goalBlocks)), agent_name)
                    self._phase = Phase.UPDATE_GOAL_LIST
                    self._navigator.reset_full()
                    self._navigator.add_waypoints([self._goalBlocks[0]['location']])
                    self._checkGoals = []
                    for g in self._goalBlocks:
                        self._checkGoals.append(g)
                elif len(self._goalBlocks) == 0:
                    self._goalBlocks = state.get_with_property({'is_goal_block': True})
                    self._phase = Phase.CHECK_GOALS

                self._sendMessage('Dropped goal block {\"size\": ' + str(
                                        self._carryingO['visualization']['size']) + ', \"shape\": ' + str(
                                        self._carryingO['visualization']['shape']) + ', \"colour\": ' + str(
                                        self._carryingO['visualization']['colour']) + '} at location ' + str(self.state.get_self()['location']),
                                  agent_name)

                self._carrying = None
                self._carryingO = None
                return DropObject.__name__, {}

            if Phase.CHECK_GOALS == self._phase:
                self._phase = Phase.FOLLOW_PATH_TO_GOAL
                if len(self._goalsWrong) != 0:
                    if len(self._possibleGoalBLocks) == 0:
                        self._phase = Phase.PLAN_PATH_TO_ROOM
                    else:
                        block = self._possibleGoalBLocks[0]
                        self._navigator.reset_full()
                        self._navigator.add_waypoints([block['location']])
                        self._phase = Phase.MOVING_TO_KNOWN_BLOCK
                    return None, {}
                if len(self._goalBlocks) == 0:
                    if len(self._possibleGoalBLocks) == 0:
                        self._phase = Phase.PLAN_PATH_TO_ROOM
                    else:
                        block = self._possibleGoalBLocks[0]
                        self._navigator.reset_full()
                        self._navigator.add_waypoints([block['location']])
                        self._phase = Phase.MOVING_TO_KNOWN_BLOCK
                    return None, {}
                goal = self._goalBlocks[0]
                self._navigator.reset_full()
                self._navigator.add_waypoints([goal['location']])

            if Phase.FOLLOW_PATH_TO_GOAL == self._phase:
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    return action, {}
                self._phase = Phase.PICK_UP_GOAL_BLOCK

            if Phase.PICK_UP_GOAL_BLOCK == self._phase:
                objects = state.get_closest_with_property({'class_inheritance': ['CollectableBlock']})
                if objects is not None:
                    for o in objects:
                        if o['location'] == self.state.get_self()['location']:
                            if o['visualization']['shape'] != self._goalBlocks[0]['visualization']['shape'] or o['visualization']['colour'] != self._goalBlocks[0]['visualization']['colour']:
                                self._phase = Phase.PUT_AWAY_WRONG_BLOCK
                                self._navigator.reset_full()
                                self._navigator.add_waypoints([[self._goalBlocks[0]['location'][0], self._goalBlocks[0]['location'][1] - 3]])
                            else:
                                self._carryingO = o
                                self._phase = Phase.MOVE_GOAL_BLOCK
                            self._sendMessage('Picking up goal block {\"size\": ' + str(
                                o['visualization']['size']) + ', \"shape\": ' + str(
                                o['visualization']['shape']) + ', \"colour\": ' + str(
                                o['visualization']['colour']) + '} at location ' + str(
                                o['location']), agent_name)
                            action = GrabObject.__name__
                            self._carryingO = o
                            action_kwargs = {}
                            action_kwargs['object_id'] = o['obj_id']
                            return action, action_kwargs
                if len(self._possibleGoalBLocks) == 0:
                    self._phase = Phase.PLAN_PATH_TO_ROOM
                else:
                    block = self._possibleGoalBLocks[0]
                    self._navigator.reset_full()
                    self._navigator.add_waypoints([block['location']])
                    self._phase = Phase.MOVING_TO_KNOWN_BLOCK

            if Phase.PUT_AWAY_WRONG_BLOCK == self._phase:
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    return action, {}
                self._goalsWrong.append(self._goalBlocks[0])

                self._phase = Phase.CHECK_GOALS

                return DropObject.__name__, {}

            if Phase.MOVE_GOAL_BLOCK == self._phase:
                self._phase = Phase.CHECK_GOALS
                self._goalBlocks.remove(self._goalBlocks[0])

                self._sendMessage('Dropped goal block {\"size\": ' + str(
                                        self._carryingO['visualization']['size']) + ', \"shape\": ' + str(
                                        self._carryingO['visualization']['shape']) + ', \"colour\": ' + str(
                                        self._carryingO['visualization']['colour']) + '} at location ' + str(
                    self.state.get_self()['location']),
                                  agent_name)
                self._carryingO = None
                return DropObject.__name__, {}

            if Phase.UPDATE_GOAL_LIST == self._phase:
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    return action, {}
                for g in self._goalBlocks:
                    self._sendMessage('Goal block {\"size\": ' + str(
                        g['visualization']['size']) + ', \"shape\": ' + str(
                        g['visualization']['shape']) + ', \"colour\": ' + str(
                        g['visualization']['colour']) + '} at location ' + str(
                        self.state.get_self()['location']),
                                      agent_name)
                if len(self._goalBlocks) == 0:
                    self._goalBlocks = state.get_with_property({'is_goal_block': True})
                    self._phase = Phase.CHECK_GOALS
                    return None, {}
                # If there is a block on the goal, update the goallist
                if len(self._checkGoals) == 0:
                    self._goalBlocks = state.get_with_property({'is_goal_block': True})
                    self._phase = Phase.CHECK_GOALS
                    return None, {}
                goal = self._checkGoals[0]
                objects = state.get_closest_with_property({'class_inheritance': ['CollectableBlock']})
                if objects is not None:
                    for o in objects:
                        if o['location'] == goal['location']:
                            self._goalBlocks.remove(goal)
                            if len(self._goalBlocks) == 0:
                                self._goalBlocks = state.get_with_property({'is_goal_block': True})
                                self._phase = Phase.CHECK_GOALS
                                return None, {}
                            elif len(self._checkGoals) == 0:
                                if len(self._possibleGoalBLocks) == 0:
                                    self._phase = Phase.PLAN_PATH_TO_ROOM
                                else:
                                    block = self._possibleGoalBLocks[0]
                                    self._navigator.reset_full()
                                    self._navigator.add_waypoints([block['location']])
                                    self._phase = Phase.MOVING_TO_KNOWN_BLOCK
                            else:
                                next = self._checkGoals[0]
                                self._navigator.reset_full()
                                self._navigator.add_waypoints([next['location']])
                self._checkGoals.remove(goal)

            # TODO: Has not been tested, since it does not parse messages yet
            if Phase.MOVING_TO_KNOWN_BLOCK == self._phase:
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    return action, {}
                objects = state.get_closest_with_property({'class_inheritance': ['CollectableBlock']})
                if objects is not None:
                    for o in objects:
                        if o['location'] == self._possibleGoalBLocks[0]['location'] and o['visualization'] == self._possibleGoalBLocks[0]['visualization']:
                            for g in self._goalBlocks:
                                if o['visualization']['shape'] == g['visualization']['shape'] and o['visualization']['colour'] == g['visualization']['colour']:
                                    self._sendMessage('Found goal block {\"size\": ' + str(
                                        o['visualization']['size']) + ', \"shape\": ' + str(
                                        o['visualization']['shape']) + ', \"colour\": ' + str(
                                        o['visualization']['colour']) + '} at location ' + str(o['location']),
                                                      agent_name)
                                    self._sendMessage('Picking up goal block {\"size\": ' + str(
                                        o['visualization']['size']) + ', \"shape\": ' + str(
                                        o['visualization']['shape']) + ', \"colour\": ' + str(
                                        o['visualization']['colour']) + '} at location ' + str(
                                        o['location']), agent_name)
                                    self._phase = Phase.FOLLOW_PATH_TO_DROP
                                    self._navigator.reset_full()
                                    self._navigator.add_waypoints([g['location']])
                                    action = GrabObject.__name__
                                    action_kwargs = {}
                                    action_kwargs['object_id'] = o['obj_id']
                                    self._carrying = g
                                    self._carryingO = o
                                    return action, action_kwargs
                self._possibleGoalBLocks.remove(self._possibleGoalBLocks[0])
                if len(self._possibleGoalBLocks) == 0:
                    self._phase = Phase.PLAN_PATH_TO_ROOM
                else:
                    block = self._possibleGoalBLocks[0]
                    self._navigator.reset_full()
                    self._navigator.add_waypoints([block['location']])
                    self._phase = Phase.MOVING_TO_KNOWN_BLOCK


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

class LiarAgent(BW4TBrain):

    def __init__(self, settings: Dict[str, object]):
        super().__init__(settings)
        self._phase = Phase.PLAN_PATH_TO_ROOM
        self._teamMembers = []

    def initialize(self):
        super().initialize()
        self._state_tracker = StateTracker(agent_id=self.agent_id)
        self._navigator = Navigator(agent_id=self.agent_id,
                                    action_set=self.action_set, algorithm=Navigator.A_STAR_ALGORITHM)
        self._goalBlocks = None
        self._goalsInitialized = False
        self._carrying = None
        self._carryingO = None
        self._goalsWrong = []
        self._checkGoals = []
        self._possibleGoalBLocks = []
        self._notExplored = []

    def filter_bw4t_observations(self, state):
        return state

    def decide_on_bw4t_action(self, state: State):
        if not self._goalsInitialized:
            self._goalBlocks = state.get_with_property({'is_goal_block': True})
            self._goalsInitialized = True
            self._notExplored = [door for door in state.values()
                               if 'class_inheritance' in door and 'Door' in door['class_inheritance']]

        agent_name = state[self.agent_id]['obj_id']
        # Add team members
        for member in state['World']['team_members']:
            if member != agent_name and member not in self._teamMembers:
                self._teamMembers.append(member)
                # Process messages from team members
        receivedMessages = self._processMessages(self._teamMembers)
        # Update trust beliefs for team members
        self._trustBlief(self._teamMembers, receivedMessages)

        objects = state.get_closest_with_property({'class_inheritance': ['CollectableBlock']})
        if objects is not None:
            for o in objects:
                count = 0
                for g in self._goalBlocks:
                    if o['location'] == g['location']:
                        count += 1
                if count == 0:
                    for g in self._goalBlocks:
                        if o['visualization']['shape'] == g['visualization']['shape'] and o['visualization'][
                            'colour'] == \
                                g['visualization']['colour'] and len(o['carried_by']) == 0:
                            self._possibleGoalBLocks.append(o)
                            self._sendMessage('Found goal block {\"size\": ' + str(
                                o['visualization']['size']) + ', \"shape\": ' + str(
                                o['visualization']['shape']) + ', \"colour\": ' + str(
                                o['visualization']['colour']) + '} at location ' + str(o['location']), agent_name)

        while True:
            if Phase.PLAN_PATH_TO_ROOM == self._phase:
                self._navigator.reset_full()
                rooms = [door for door in state.values()
                               if 'class_inheritance' in door and 'Door' in door['class_inheritance']]
                if len(rooms) == 0:
                    return None, {}
                # If not every room has been explored, pick one randomly from the non explored, else pick randomly pick a door
                if len(self._notExplored) != 0:
                    self._door = random.choice(self._notExplored)
                else:
                    self._door = random.choice(rooms)
                doorLoc = self._door['location']
                # Location in front of door is south from door
                doorLoc = doorLoc[0], doorLoc[1] + 1
                # Send message of current action
                self._sendMessage('Moving to ' + self._door['room_name'], agent_name, state)
                self._navigator.add_waypoints([doorLoc])
                self._phase = Phase.FOLLOW_PATH_TO_ROOM

            if Phase.FOLLOW_PATH_TO_ROOM == self._phase:
                self._state_tracker.update(state)
                # Follow path to room
                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    return action, {}
                if self._door['is_open']:
                    self._phase = Phase.ENTERING_ROOM
                else:
                     self._phase = Phase.OPEN_DOOR

            if Phase.OPEN_DOOR == self._phase:
                self._phase = Phase.WAIT_FOR_DOOR
                # Open door
                self._sendMessage('Opening door of ' + self._door['room_name'], agent_name, state)
                return OpenDoorAction.__name__, {'object_id': self._door['obj_id']}

            if Phase.WAIT_FOR_DOOR == self._phase:
                self._phase = Phase.ENTERING_ROOM
                return None, {}

            if Phase.ENTERING_ROOM == self._phase:
                if self._door in self._notExplored:
                    self._notExplored.remove(self._door)
                self._sendMessage('Searching through ' + self._door['room_name'], agent_name, state)
                self._navigator.reset_full()
                objects = state.get_room_objects(self._door['room_name'])
                for o in objects:
                    if o['is_traversable']:
                        self._navigator.add_waypoints([o['location']])
                self._phase = Phase.SEARCHING_ROOM

            if Phase.SEARCHING_ROOM == self._phase:
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                objects = state.get_closest_with_property({'class_inheritance': ['CollectableBlock']})
                if objects is not None:
                    for o in objects:
                        for g in self._goalBlocks:
                            if o['visualization']['shape'] == g['visualization']['shape'] and o['visualization']['colour'] == g['visualization']['colour'] and len(o['carried_by']) == 0:
                                self._sendMessage('Found goal block {\"size\": ' + str(
                                    o['visualization']['size']) + ', \"shape\": ' + str(
                                    o['visualization']['shape']) + ', \"colour\": ' + str(
                                    o['visualization']['colour']) + '} at location ' + str(o['location']), agent_name, state)
                                self._sendMessage('Picking up goal block {\"size\": ' + str(
                                    o['visualization']['size']) + ', \"shape\": ' + str(
                                    o['visualization']['shape']) + ', \"colour\": ' + str(
                                    o['visualization']['colour']) + '} at location ' + str(
                                    o['location']), agent_name, state)
                                self._phase = Phase.FOLLOW_PATH_TO_DROP
                                self._navigator.reset_full()
                                self._navigator.add_waypoints([g['location']])
                                action = GrabObject.__name__
                                action_kwargs = {}
                                action_kwargs['object_id'] = o['obj_id']
                                self._carrying = g
                                self._carryingO = o
                                return action, action_kwargs
                if action!=None:
                    return action, {}
                if len(self._possibleGoalBLocks) == 0:
                    self._phase = Phase.PLAN_PATH_TO_ROOM
                else:
                    block = self._possibleGoalBLocks[0]
                    self._navigator.reset_full()
                    self._navigator.add_waypoints([block['location']])
                    self._phase = Phase.MOVING_TO_KNOWN_BLOCK

            if Phase.FOLLOW_PATH_TO_DROP == self._phase:
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    return action, {}
                self._phase = Phase.DROP_OBJECT

            if Phase.DROP_OBJECT == self._phase:
                if len(self._possibleGoalBLocks) == 0:
                    self._phase = Phase.PLAN_PATH_TO_ROOM
                else:
                    block = self._possibleGoalBLocks[0]
                    self._navigator.reset_full()
                    self._navigator.add_waypoints([block['location']])
                    self._phase = Phase.MOVING_TO_KNOWN_BLOCK

                # If there already is a block in this location, move to a different location and drop the block there.
                objects = state.get_closest_with_property({'class_inheritance': ['CollectableBlock']})
                if objects is not None:
                    for o in objects:
                        if o['location'] == self.state.get_self()['location']:
                            self._navigator.reset_full()
                            self._navigator.add_waypoints([self._carryingO['location']])
                            self._phase = Phase.FOLLOW_PATH_TO_DROP
                            return None, {}

                self._goalBlocks.remove(self._carrying)
                self._sendMessage('Removing block' + str(self._carrying) + ' from list: ' + str(self._goalBlocks),
                                  agent_name, state)

                if len(self._goalBlocks) >= 1:
                    self._sendMessage('Updating goal list with ' + str(len(self._goalBlocks)), agent_name, state)
                    self._phase = Phase.UPDATE_GOAL_LIST
                    self._navigator.reset_full()
                    self._navigator.add_waypoints([self._goalBlocks[0]['location']])
                    self._checkGoals = []
                    for g in self._goalBlocks:
                        self._checkGoals.append(g)
                elif len(self._goalBlocks) == 0:
                    self._goalBlocks = state.get_with_property({'is_goal_block': True})
                    self._phase = Phase.CHECK_GOALS

                self._sendMessage('Dropped goal block {\"size\": ' + str(
                                        self._carryingO['visualization']['size']) + ', \"shape\": ' + str(
                                        self._carryingO['visualization']['shape']) + ', \"colour\": ' + str(
                                        self._carryingO['visualization']['colour']) + '} at location ' + str(self.state.get_self()['location']),
                                  agent_name, state)

                self._carrying = None
                self._carryingO = None
                return DropObject.__name__, {}

            if Phase.CHECK_GOALS == self._phase:
                self._phase = Phase.FOLLOW_PATH_TO_GOAL
                if len(self._goalsWrong) != 0:
                    if len(self._possibleGoalBLocks) == 0:
                        self._phase = Phase.PLAN_PATH_TO_ROOM
                    else:
                        block = self._possibleGoalBLocks[0]
                        self._navigator.reset_full()
                        self._navigator.add_waypoints([block['location']])
                        self._phase = Phase.MOVING_TO_KNOWN_BLOCK
                    return None, {}
                if len(self._goalBlocks) == 0:
                    if len(self._possibleGoalBLocks) == 0:
                        self._phase = Phase.PLAN_PATH_TO_ROOM
                    else:
                        block = self._possibleGoalBLocks[0]
                        self._navigator.reset_full()
                        self._navigator.add_waypoints([block['location']])
                        self._phase = Phase.MOVING_TO_KNOWN_BLOCK
                    return None, {}
                goal = self._goalBlocks[0]
                self._navigator.reset_full()
                self._navigator.add_waypoints([goal['location']])

            if Phase.FOLLOW_PATH_TO_GOAL == self._phase:
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    return action, {}
                self._phase = Phase.PICK_UP_GOAL_BLOCK

            if Phase.PICK_UP_GOAL_BLOCK == self._phase:
                objects = state.get_closest_with_property({'class_inheritance': ['CollectableBlock']})
                if objects is not None:
                    for o in objects:
                        if o['location'] == self.state.get_self()['location']:
                            if o['visualization']['shape'] != self._goalBlocks[0]['visualization']['shape'] or o['visualization']['colour'] != self._goalBlocks[0]['visualization']['colour']:
                                self._phase = Phase.PUT_AWAY_WRONG_BLOCK
                                self._navigator.reset_full()
                                self._navigator.add_waypoints([[self._goalBlocks[0]['location'][0], self._goalBlocks[0]['location'][1] - 3]])
                            else:
                                self._carryingO = o
                                self._phase = Phase.MOVE_GOAL_BLOCK
                            self._sendMessage('Picking up goal block {\"size\": ' + str(
                                o['visualization']['size']) + ', \"shape\": ' + str(
                                o['visualization']['shape']) + ', \"colour\": ' + str(
                                o['visualization']['colour']) + '} at location ' + str(
                                o['location']), agent_name, state)
                            action = GrabObject.__name__
                            self._carryingO = o
                            action_kwargs = {}
                            action_kwargs['object_id'] = o['obj_id']
                            return action, action_kwargs
                if len(self._possibleGoalBLocks) == 0:
                    self._phase = Phase.PLAN_PATH_TO_ROOM
                else:
                    block = self._possibleGoalBLocks[0]
                    self._navigator.reset_full()
                    self._navigator.add_waypoints([block['location']])
                    self._phase = Phase.MOVING_TO_KNOWN_BLOCK

            if Phase.PUT_AWAY_WRONG_BLOCK == self._phase:
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    return action, {}
                self._goalsWrong.append(self._goalBlocks[0])

                self._phase = Phase.CHECK_GOALS

                return DropObject.__name__, {}

            if Phase.MOVE_GOAL_BLOCK == self._phase:
                self._phase = Phase.CHECK_GOALS
                self._goalBlocks.remove(self._goalBlocks[0])

                self._sendMessage('Dropped goal block {\"size\": ' + str(
                                        self._carryingO['visualization']['size']) + ', \"shape\": ' + str(
                                        self._carryingO['visualization']['shape']) + ', \"colour\": ' + str(
                                        self._carryingO['visualization']['colour']) + '} at location ' + str(
                    self.state.get_self()['location']),
                                  agent_name, state)
                self._carryingO = None
                return DropObject.__name__, {}

            if Phase.UPDATE_GOAL_LIST == self._phase:
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    return action, {}
                for g in self._goalBlocks:
                    self._sendMessage('Goal block {\"size\": ' + str(
                        g['visualization']['size']) + ', \"shape\": ' + str(
                        g['visualization']['shape']) + ', \"colour\": ' + str(
                        g['visualization']['colour']) + '} at location ' + str(
                        self.state.get_self()['location']),
                                      agent_name, state)
                if len(self._goalBlocks) == 0:
                    self._goalBlocks = state.get_with_property({'is_goal_block': True})
                    self._phase = Phase.CHECK_GOALS
                    return None, {}
                # If there is a block on the goal, update the goallist
                if len(self._checkGoals) == 0:
                    self._goalBlocks = state.get_with_property({'is_goal_block': True})
                    self._phase = Phase.CHECK_GOALS
                    return None, {}
                goal = self._checkGoals[0]
                objects = state.get_closest_with_property({'class_inheritance': ['CollectableBlock']})
                if objects is not None:
                    for o in objects:
                        if o['location'] == goal['location']:
                            self._goalBlocks.remove(goal)
                            if len(self._goalBlocks) == 0:
                                self._goalBlocks = state.get_with_property({'is_goal_block': True})
                                self._phase = Phase.CHECK_GOALS
                                return None, {}
                            elif len(self._checkGoals) == 0:
                                if len(self._possibleGoalBLocks) == 0:
                                    self._phase = Phase.PLAN_PATH_TO_ROOM
                                else:
                                    block = self._possibleGoalBLocks[0]
                                    self._navigator.reset_full()
                                    self._navigator.add_waypoints([block['location']])
                                    self._phase = Phase.MOVING_TO_KNOWN_BLOCK
                            else:
                                next = self._checkGoals[0]
                                self._navigator.reset_full()
                                self._navigator.add_waypoints([next['location']])
                self._checkGoals.remove(goal)

            # TODO: Has not been tested, since it does not parse messages yet
            if Phase.MOVING_TO_KNOWN_BLOCK == self._phase:
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    return action, {}
                objects = state.get_closest_with_property({'class_inheritance': ['CollectableBlock']})
                if objects is not None:
                    for o in objects:
                        if o['location'] == self._possibleGoalBLocks[0]['location'] and o['visualization'] == self._possibleGoalBLocks[0]['visualization']:
                            for g in self._goalBlocks:
                                if o['visualization']['shape'] == g['visualization']['shape'] and o['visualization']['colour'] == g['visualization']['colour']:
                                    self._sendMessage('Found goal block {\"size\": ' + str(
                                        o['visualization']['size']) + ', \"shape\": ' + str(
                                        o['visualization']['shape']) + ', \"colour\": ' + str(
                                        o['visualization']['colour']) + '} at location ' + str(o['location']),
                                                      agent_name, state)
                                    self._sendMessage('Picking up goal block {\"size\": ' + str(
                                        o['visualization']['size']) + ', \"shape\": ' + str(
                                        o['visualization']['shape']) + ', \"colour\": ' + str(
                                        o['visualization']['colour']) + '} at location ' + str(
                                        o['location']), agent_name, state)
                                    self._phase = Phase.FOLLOW_PATH_TO_DROP
                                    self._navigator.reset_full()
                                    self._navigator.add_waypoints([g['location']])
                                    action = GrabObject.__name__
                                    action_kwargs = {}
                                    action_kwargs['object_id'] = o['obj_id']
                                    self._carrying = g
                                    self._carryingO = o
                                    return action, action_kwargs
                self._possibleGoalBLocks.remove(self._possibleGoalBLocks[0])
                if len(self._possibleGoalBLocks) == 0:
                    self._phase = Phase.PLAN_PATH_TO_ROOM
                else:
                    block = self._possibleGoalBLocks[0]
                    self._navigator.reset_full()
                    self._navigator.add_waypoints([block['location']])
                    self._phase = Phase.MOVING_TO_KNOWN_BLOCK


    def _sendMessage(self, mssg, sender, state):
        '''
        Enable sending messages in one line of code
        '''
        msssg = mssg
        if random.random() < 0.8:
            msssg = self._generateLie(state)
        msg = Message(content=msssg, from_id=sender)
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

    def _generateLie(self, state: State):
        rand = random.randint(1,6)
        if rand == 1:
            choices = self._getRoomNames(state)
            return 'Moving to ' + random.choice(choices)
        if rand == 2:
            choices = self._getRoomNames(state)
            return 'Searching through ' + random.choice(choices)
        if rand == 3:
            choices = self._getRoomNames(state)
            return 'Opening door of ' + random.choice(choices)
        if rand == 4:
            o = random.choice(self._goalBlocks)
            memyself = state.get_self()
            return 'Found goal block {\"size\": ' + str(
                                    o['visualization']['size']) + ', \"shape\": ' + str(
                                    o['visualization']['shape']) + ', \"colour\": ' + str(
                                    o['visualization']['colour']) + '} at location ' + str(memyself['location'])
        if rand == 5:
            o = random.choice(self._goalBlocks)
            memyself = state.get_self()
            return 'Picking up goal block {\"size\": ' + str(
                o['visualization']['size']) + ', \"shape\": ' + str(
                o['visualization']['shape']) + ', \"colour\": ' + str(
                o['visualization']['colour']) + '} at location ' + str(
                memyself['location'])
        if rand == 6:
            o = random.choice(self._goalBlocks)
            return 'Dropped goal block {\"size\": ' + str(
                                        o['visualization']['size']) + ', \"shape\": ' + str(
                                        o['visualization']['shape']) + ', \"colour\": ' + str(
                                        o['visualization']['colour']) + '} at location ' + str(
                                        o['location'])



    def _getRoomNames(self, state: State):
        room_names = state.get_all_room_names()
        if 'world_bounds' in room_names:
            room_names.remove('world_bounds')
        return room_names

class LazyAgent(BW4TBrain):

    def __init__(self, settings: Dict[str, object]):
        super().__init__(settings)
        self._phase = Phase.PLAN_PATH_TO_ROOM
        self._teamMembers = []

    def initialize(self):
        super().initialize()
        self._state_tracker = StateTracker(agent_id=self.agent_id)
        self._navigator = Navigator(agent_id=self.agent_id,
                                    action_set=self.action_set, algorithm=Navigator.A_STAR_ALGORITHM)
        self._goalBlocks = None
        self._goalsInitialized = False
        self._carrying = None
        self._carryingO = None
        self._lazymoment = False
        self._prevloc = None
        self._goalsWrong = []
        self._checkGoals = []
        self._possibleGoalBLocks = []
        self._notExplored = []

    def filter_bw4t_observations(self, state):
        return state

    def decide_on_bw4t_action(self, state: State):
        if not self._goalsInitialized:
            self._goalBlocks = state.get_with_property({'is_goal_block': True})
            self._goalsInitialized = True
            self._notExplored = [door for door in state.values()
                               if 'class_inheritance' in door and 'Door' in door['class_inheritance']]

        agent_name = state[self.agent_id]['obj_id']
        # Add team members
        for member in state['World']['team_members']:
            if member != agent_name and member not in self._teamMembers:
                self._teamMembers.append(member)
                # Process messages from team members
        receivedMessages = self._processMessages(self._teamMembers)
        # Update trust beliefs for team members
        self._trustBlief(self._teamMembers, receivedMessages)

        objects = state.get_closest_with_property({'class_inheritance': ['CollectableBlock']})
        if objects is not None:
            for o in objects:
                count = 0
                for g in self._goalBlocks:
                    if o['location'] == g['location']:
                        count += 1
                if count == 0:
                    for g in self._goalBlocks:
                        if o['visualization']['shape'] == g['visualization']['shape'] and o['visualization'][
                            'colour'] == \
                                g['visualization']['colour'] and len(o['carried_by']) == 0:
                            self._possibleGoalBLocks.append(o)
                            self._sendMessage('Found goal block {\"size\": ' + str(
                                o['visualization']['size']) + ', \"shape\": ' + str(
                                o['visualization']['shape']) + ', \"colour\": ' + str(
                                o['visualization']['colour']) + '} at location ' + str(o['location']), agent_name)


        while True:
            if Phase.PLAN_PATH_TO_ROOM == self._phase:
                self._navigator.reset_full()
                rooms = [door for door in state.values()
                               if 'class_inheritance' in door and 'Door' in door['class_inheritance']]
                if len(rooms) == 0:
                    return None, {}
                # If not every room has been explored, pick one randomly from the non explored, else pick randomly pick a door
                if len(self._notExplored) != 0:
                    self._door = random.choice(self._notExplored)
                else:
                    self._door = random.choice(rooms)
                doorLoc = self._door['location']
                # Location in front of door is south from door
                doorLoc = doorLoc[0], doorLoc[1] + 1
                # Send message of current action
                self._sendMessage('Moving to ' + self._door['room_name'], agent_name)
                self._navigator.add_waypoints([doorLoc])

                self._lazymoment = False
                if random.random() < 0.5:
                    self._lazymoment = True

                self._phase = Phase.FOLLOW_PATH_TO_ROOM

            if Phase.FOLLOW_PATH_TO_ROOM == self._phase:
                self._state_tracker.update(state)
                # Follow path to room
                if self._lazymoment and random.random() < 0.07:
                    self._phase = Phase.PLAN_PATH_TO_ROOM
                    return None, {}

                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    return action, {}
                if self._lazymoment:
                    self._phase = Phase.PLAN_PATH_TO_ROOM
                    return None, {}
                if self._door['is_open']:
                    self._phase = Phase.ENTERING_ROOM
                else:
                     self._phase = Phase.OPEN_DOOR

            if Phase.OPEN_DOOR == self._phase:
                self._phase = Phase.WAIT_FOR_DOOR
                # Open door
                self._sendMessage('Opening door of ' + self._door['room_name'], agent_name)
                if random.random() < 0.5:
                    self._phase = Phase.PLAN_PATH_TO_ROOM
                    return None, {}
                return OpenDoorAction.__name__, {'object_id': self._door['obj_id']}

            if Phase.WAIT_FOR_DOOR == self._phase:
                self._phase = Phase.ENTERING_ROOM
                return None, {}

            if Phase.ENTERING_ROOM == self._phase:
                if self._door in self._notExplored:
                    self._notExplored.remove(self._door)
                self._sendMessage('Searching through ' + self._door['room_name'], agent_name)
                self._navigator.reset_full()
                objects = state.get_room_objects(self._door['room_name'])
                for o in objects:
                    if o['is_traversable']:
                        self._navigator.add_waypoints([o['location']])

                self._lazymoment = False
                if random.random() < 0.5:
                    self._lazymoment = True

                self._phase = Phase.SEARCHING_ROOM

            if Phase.SEARCHING_ROOM == self._phase:
                self._state_tracker.update(state)
                if self._lazymoment and random.random() < 0.15:
                    self._phase = Phase.PLAN_PATH_TO_ROOM

                action = self._navigator.get_move_action(self._state_tracker)
                objects = state.get_closest_with_property({'class_inheritance': ['CollectableBlock']})
                if objects is not None:
                    for o in objects:
                        for g in self._goalBlocks:
                            if o['visualization']['shape'] == g['visualization']['shape'] and o['visualization']['colour'] == g['visualization']['colour'] and len(o['carried_by']) == 0:
                                self._sendMessage('Found goal block {\"size\": ' + str(
                                    o['visualization']['size']) + ', \"shape\": ' + str(
                                    o['visualization']['shape']) + ', \"colour\": ' + str(
                                    o['visualization']['colour']) + '} at location ' + str(o['location']), agent_name)
                                self._sendMessage('Picking up goal block {\"size\": ' + str(
                                    o['visualization']['size']) + ', \"shape\": ' + str(
                                    o['visualization']['shape']) + ', \"colour\": ' + str(
                                    o['visualization']['colour']) + '} at location ' + str(
                                    o['location']), agent_name)
                                if self._lazymoment:
                                    self._phase = Phase.PLAN_PATH_TO_ROOM
                                    return None, {}
                                self._phase = Phase.FOLLOW_PATH_TO_DROP
                                self._navigator.reset_full()
                                self._navigator.add_waypoints([g['location']])
                                action = GrabObject.__name__
                                action_kwargs = {}
                                action_kwargs['object_id'] = o['obj_id']
                                self._carrying = g
                                self._carryingO = o
                                self._lazymoment = False
                                self._prevloc = None
                                if random.random() < 0.5:
                                    self._lazymoment = True
                                return action, action_kwargs
                if action!=None:
                    return action, {}

                if len(self._possibleGoalBLocks) == 0 or self._lazymoment:
                    self._phase = Phase.PLAN_PATH_TO_ROOM
                else:
                    block = self._possibleGoalBLocks[0]
                    self._navigator.reset_full()
                    self._navigator.add_waypoints([block['location']])
                    self._phase = Phase.MOVING_TO_KNOWN_BLOCK

            if Phase.FOLLOW_PATH_TO_DROP == self._phase:
                self._state_tracker.update(state)
                if self._lazymoment and random.random() < 0.07:
                    self._phase = Phase.DROP_OBJECT
                    return None, {}
                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    self._prevloc = state.get_self()['location']
                    return action, {}
                if self._lazymoment and self._prevloc is not None:
                    self._phase = Phase.RUN_BACK
                    self._navigator.reset_full()
                    self._navigator.add_waypoints([self._prevloc])
                    return None, {}
                self._phase = Phase.DROP_OBJECT

            if Phase.RUN_BACK == self._phase:
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    return action, {}
                self._phase = Phase.DROP_OBJECT

            if Phase.DROP_OBJECT == self._phase:
                if len(self._possibleGoalBLocks) == 0:
                    self._phase = Phase.PLAN_PATH_TO_ROOM
                else:
                    block = self._possibleGoalBLocks[0]
                    self._navigator.reset_full()
                    self._navigator.add_waypoints([block['location']])
                    self._phase = Phase.MOVING_TO_KNOWN_BLOCK

                # If there already is a block in this location, move to a different location and drop the block there.
                objects = state.get_closest_with_property({'class_inheritance': ['CollectableBlock']})
                if objects is not None:
                    for o in objects:
                        if o['location'] == self.state.get_self()['location']:
                            self._navigator.reset_full()
                            self._navigator.add_waypoints([self._carryingO['location']])
                            self._phase = Phase.FOLLOW_PATH_TO_DROP
                            return None, {}

                self._goalBlocks.remove(self._carrying)
                self._sendMessage('Removing block' + str(self._carrying) + ' from list: ' + str(self._goalBlocks),
                                  agent_name)

                if len(self._goalBlocks) >= 1:
                    self._sendMessage('Updating goal list with ' + str(len(self._goalBlocks)), agent_name)
                    self._phase = Phase.UPDATE_GOAL_LIST
                    self._navigator.reset_full()
                    self._navigator.add_waypoints([self._goalBlocks[0]['location']])
                    self._checkGoals = []
                    for g in self._goalBlocks:
                        self._checkGoals.append(g)
                elif len(self._goalBlocks) == 0:
                    self._goalBlocks = state.get_with_property({'is_goal_block': True})
                    self._phase = Phase.CHECK_GOALS

                self._sendMessage('Dropped goal block {\"size\": ' + str(
                                        self._carryingO['visualization']['size']) + ', \"shape\": ' + str(
                                        self._carryingO['visualization']['shape']) + ', \"colour\": ' + str(
                                        self._carryingO['visualization']['colour']) + '} at location ' + str(self.state.get_self()['location']),
                                  agent_name)

                self._carrying = None
                self._carryingO = None
                return DropObject.__name__, {}

            if Phase.CHECK_GOALS == self._phase:
                self._phase = Phase.FOLLOW_PATH_TO_GOAL
                if len(self._goalsWrong) != 0:
                    if len(self._possibleGoalBLocks) == 0:
                        self._phase = Phase.PLAN_PATH_TO_ROOM
                    else:
                        block = self._possibleGoalBLocks[0]
                        self._navigator.reset_full()
                        self._navigator.add_waypoints([block['location']])
                        self._phase = Phase.MOVING_TO_KNOWN_BLOCK
                    return None, {}
                if len(self._goalBlocks) == 0:
                    if len(self._possibleGoalBLocks) == 0:
                        self._phase = Phase.PLAN_PATH_TO_ROOM
                    else:
                        block = self._possibleGoalBLocks[0]
                        self._navigator.reset_full()
                        self._navigator.add_waypoints([block['location']])
                        self._phase = Phase.MOVING_TO_KNOWN_BLOCK
                    return None, {}
                goal = self._goalBlocks[0]
                self._navigator.reset_full()
                self._navigator.add_waypoints([goal['location']])

            if Phase.FOLLOW_PATH_TO_GOAL == self._phase:
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    return action, {}
                self._phase = Phase.PICK_UP_GOAL_BLOCK

            if Phase.PICK_UP_GOAL_BLOCK == self._phase:
                objects = state.get_closest_with_property({'class_inheritance': ['CollectableBlock']})
                if objects is not None:
                    for o in objects:
                        if o['location'] == self.state.get_self()['location']:
                            if o['visualization']['shape'] != self._goalBlocks[0]['visualization']['shape'] or o['visualization']['colour'] != self._goalBlocks[0]['visualization']['colour']:
                                self._phase = Phase.PUT_AWAY_WRONG_BLOCK
                                self._navigator.reset_full()
                                self._navigator.add_waypoints([[self._goalBlocks[0]['location'][0], self._goalBlocks[0]['location'][1] - 3]])
                            else:
                                self._carryingO = o
                                self._phase = Phase.MOVE_GOAL_BLOCK
                            self._sendMessage('Picking up goal block {\"size\": ' + str(
                                o['visualization']['size']) + ', \"shape\": ' + str(
                                o['visualization']['shape']) + ', \"colour\": ' + str(
                                o['visualization']['colour']) + '} at location ' + str(
                                o['location']), agent_name)
                            action = GrabObject.__name__
                            self._carryingO = o
                            action_kwargs = {}
                            action_kwargs['object_id'] = o['obj_id']
                            return action, action_kwargs
                if len(self._possibleGoalBLocks) == 0:
                    self._phase = Phase.PLAN_PATH_TO_ROOM
                else:
                    block = self._possibleGoalBLocks[0]
                    self._navigator.reset_full()
                    self._navigator.add_waypoints([block['location']])
                    self._phase = Phase.MOVING_TO_KNOWN_BLOCK

            if Phase.PUT_AWAY_WRONG_BLOCK == self._phase:
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    return action, {}
                self._goalsWrong.append(self._goalBlocks[0])

                self._phase = Phase.CHECK_GOALS

                return DropObject.__name__, {}

            if Phase.MOVE_GOAL_BLOCK == self._phase:
                self._phase = Phase.CHECK_GOALS
                self._goalBlocks.remove(self._goalBlocks[0])

                self._sendMessage('Dropped goal block {\"size\": ' + str(
                                        self._carryingO['visualization']['size']) + ', \"shape\": ' + str(
                                        self._carryingO['visualization']['shape']) + ', \"colour\": ' + str(
                                        self._carryingO['visualization']['colour']) + '} at location ' + str(
                    self.state.get_self()['location']),
                                  agent_name)
                self._carryingO = None
                return DropObject.__name__, {}

            if Phase.UPDATE_GOAL_LIST == self._phase:
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    return action, {}
                for g in self._goalBlocks:
                    self._sendMessage('Goal block {\"size\": ' + str(
                        g['visualization']['size']) + ', \"shape\": ' + str(
                        g['visualization']['shape']) + ', \"colour\": ' + str(
                        g['visualization']['colour']) + '} at location ' + str(
                        self.state.get_self()['location']),
                                      agent_name)
                if len(self._goalBlocks) == 0:
                    self._goalBlocks = state.get_with_property({'is_goal_block': True})
                    self._phase = Phase.CHECK_GOALS
                    return None, {}
                # If there is a block on the goal, update the goallist
                if len(self._checkGoals) == 0:
                    self._goalBlocks = state.get_with_property({'is_goal_block': True})
                    self._phase = Phase.CHECK_GOALS
                    return None, {}
                goal = self._checkGoals[0]
                objects = state.get_closest_with_property({'class_inheritance': ['CollectableBlock']})
                if objects is not None:
                    for o in objects:
                        if o['location'] == goal['location']:
                            self._goalBlocks.remove(goal)
                            if len(self._goalBlocks) == 0:
                                self._goalBlocks = state.get_with_property({'is_goal_block': True})
                                self._phase = Phase.CHECK_GOALS
                                return None, {}
                            elif len(self._checkGoals) == 0:
                                if len(self._possibleGoalBLocks) == 0:
                                    self._phase = Phase.PLAN_PATH_TO_ROOM
                                else:
                                    block = self._possibleGoalBLocks[0]
                                    self._navigator.reset_full()
                                    self._navigator.add_waypoints([block['location']])
                                    self._phase = Phase.MOVING_TO_KNOWN_BLOCK
                            else:
                                next = self._checkGoals[0]
                                self._navigator.reset_full()
                                self._navigator.add_waypoints([next['location']])
                self._checkGoals.remove(goal)

            # TODO: Has not been tested, since it does not parse messages yet
            if Phase.MOVING_TO_KNOWN_BLOCK == self._phase:
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    return action, {}
                objects = state.get_closest_with_property({'class_inheritance': ['CollectableBlock']})
                if objects is not None:
                    for o in objects:
                        if o['location'] == self._possibleGoalBLocks[0]['location'] and o['visualization'] == self._possibleGoalBLocks[0]['visualization']:
                            for g in self._goalBlocks:
                                if o['visualization']['shape'] == g['visualization']['shape'] and o['visualization']['colour'] == g['visualization']['colour']:
                                    self._sendMessage('Found goal block {\"size\": ' + str(
                                        o['visualization']['size']) + ', \"shape\": ' + str(
                                        o['visualization']['shape']) + ', \"colour\": ' + str(
                                        o['visualization']['colour']) + '} at location ' + str(o['location']),
                                                      agent_name)
                                    self._sendMessage('Picking up goal block {\"size\": ' + str(
                                        o['visualization']['size']) + ', \"shape\": ' + str(
                                        o['visualization']['shape']) + ', \"colour\": ' + str(
                                        o['visualization']['colour']) + '} at location ' + str(
                                        o['location']), agent_name)
                                    self._phase = Phase.FOLLOW_PATH_TO_DROP
                                    self._navigator.reset_full()
                                    self._navigator.add_waypoints([g['location']])
                                    action = GrabObject.__name__
                                    action_kwargs = {}
                                    action_kwargs['object_id'] = o['obj_id']
                                    self._carrying = g
                                    self._carryingO = o
                                    return action, action_kwargs
                self._possibleGoalBLocks.remove(self._possibleGoalBLocks[0])
                if len(self._possibleGoalBLocks) == 0:
                    self._phase = Phase.PLAN_PATH_TO_ROOM
                else:
                    block = self._possibleGoalBLocks[0]
                    self._navigator.reset_full()
                    self._navigator.add_waypoints([block['location']])
                    self._phase = Phase.MOVING_TO_KNOWN_BLOCK


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
