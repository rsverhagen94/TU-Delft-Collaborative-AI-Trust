import copy
import json
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
    SEARCH_ROOM = 4,
    PLAN_MOVE_IN_ROOM = 5,
    MOVE_IN_ROOM = 6,
    BRING_TO_TARGET = 7,
    PLAN_BRING_TO_TARGET = 8


class Messages(enum.Enum):
    MOVING_TO_ROOM = "Moving to ",
    OPENING_DOOR = "Opening door of ",
    SEARCHING_THROUGH = "Searching through ",
    FOUND_GOAL_BLOCK1 = "Found goal block ",
    FOUND_GOAL_BLOCK2 = " at location ",
    PICKING_UP_GOAL_BLOCK1 = "Picking up goal block ",
    PICKING_UP_GOAL_BLOCK2 = " at location ",
    DROPPED_GOAL_BLOCK1 = "Dropped goal block ",
    DROPPED_GOAL_BLOCK2 = " at drop location "


class Lazy(BW4TBrain):

    def __init__(self, settings: Dict[str, object]):
        super().__init__(settings)
        self._current_target_block = '0'
        self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR
        self._teamMembers = []
        self._missing = {}
        self._be_lazy = False
        self._be_lazy_after_moves = 0

    def initialize(self):
        super().initialize()
        self._state_tracker = StateTracker(agent_id=self.agent_id)
        self._navigator = Navigator(agent_id=self.agent_id,
                                    action_set=self.action_set, algorithm=Navigator.A_STAR_ALGORITHM)

    def filter_bw4t_observations(self, state):

        return state

    def decide_on_bw4t_action(self, state: State):
        agent_name = state[self.agent_id]['obj_id']
        # print([x for x in state.values()])
        # Add team members
        for member in state['World']['team_members']:
            if member != agent_name and member not in self._teamMembers:
                self._teamMembers.append(member)
        if not self._missing:
            # self._missing = copy.deepcopy([tile['visualization'] for tile in state.values()
            #                                if 'class_inheritance' in tile
            #                                and 'GhostBlock' in tile['class_inheritance']
            #                                ])
            # for i in range(len(self._missing)):
            #     location = (self._getTargetFromVisualization(state, self._missing[i])['location'])
            #     self._missing_locations.append(location)

            blocks = copy.deepcopy([tile['visualization'] for tile in state.values()
                                           if 'class_inheritance' in tile
                                           and 'GhostBlock' in tile['class_inheritance']
                                           ])
            for i in range(len(blocks)):
                location = (self._getTargetFromVisualization(state, blocks[i])['location'])
                self._missing[str(i)] = {'block': blocks[i], 'location': location}

            #print(self._missing)

        # Process messages from team members
        receivedMessages = self._processMessages(self._teamMembers)
        # Update trust beliefs for team members
        self._trustBlief(self._teamMembers, receivedMessages)

        while True:
            if Phase.PLAN_PATH_TO_CLOSED_DOOR == self._phase:
                self._navigator.reset_full()
                closedDoors = [door for door in state.values()
                               if 'class_inheritance' in door and 'Door' in door['class_inheritance'] and not door[
                        'is_open']]
                if len(closedDoors) == 0:
                    closedDoors = [door for door in state.values()
                                   if 'class_inheritance' in door and 'Door' in door['class_inheritance']]
                # Randomly pick a closed door
                self._door = random.choice(closedDoors)
                doorLoc = self._door['location']
                # Location in front of door is south from door
                doorLoc = doorLoc[0], doorLoc[1] + 1
                # Send message of current action
                self._sendMessage('Moving to door of ' + self._door['room_name'], agent_name, state)
                self._navigator.add_waypoints([doorLoc])
                self._set_lazy(len(self._navigator.get_all_waypoints()))
                self._phase = Phase.FOLLOW_PATH_TO_CLOSED_DOOR

            if Phase.FOLLOW_PATH_TO_CLOSED_DOOR == self._phase:
                if self._check_if_lazy():
                    self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR
                    continue
                self._state_tracker.update(state)
                # Follow path to door
                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    return action, {}
                self._phase = Phase.OPEN_DOOR

            if Phase.OPEN_DOOR == self._phase:
                self._state_tracker.update(state)
                self._sendMessage('Opening door of ' + self._door['room_name'], agent_name, state)
                self._phase = Phase.SEARCH_ROOM
                # Open door
                return OpenDoorAction.__name__, {'object_id': self._door['obj_id']}
            if Phase.SEARCH_ROOM == self._phase:
                self._room_tiles = [tile for tile in state.values()
                                    if 'class_inheritance' in tile
                                    and 'AreaTile' in tile['class_inheritance']
                                    and 'room_name' in tile
                                    and tile['room_name'] == self._door['room_name']
                                    ]
                self._sendMessage('Searching through ' + self._door['room_name'], agent_name, state)
                self._set_lazy(len(self._room_tiles))

                self._phase = Phase.PLAN_MOVE_IN_ROOM
            if Phase.PLAN_MOVE_IN_ROOM == self._phase:
                self._state_tracker.update(state)
                self._navigator.reset_full()
                if len(self._room_tiles) > 0:
                    tileLoc = self._room_tiles.pop()['location']
                    self._navigator.add_waypoints([tileLoc])
                    self._phase = Phase.MOVE_IN_ROOM
                else:
                    self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR
            if Phase.MOVE_IN_ROOM == self._phase:
                if self._check_if_lazy():
                    self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR
                    continue
                self._state_tracker.update(state)
                matching = self._getTargetBlocks(state)
                for block in matching:
                    self._sendMessage(
                        'Found goal block ' + str(block['visualization']) + ' at location ' + str(block['location']),
                        agent_name, state)
                if len(matching) > 0:
                    self._sendMessage(
                        'Picking up goal block ' + str(matching[0]['visualization']) + ' at location ' + str(
                            matching[0]['location']), agent_name, state)
                    self._phase = Phase.PLAN_BRING_TO_TARGET
                    return GrabObject.__name__, {'object_id': matching[0]['obj_id']}
                self._state_tracker.update(state)
                # Follow path to tile
                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    return action, {}
                self._phase = Phase.PLAN_MOVE_IN_ROOM
            if Phase.PLAN_BRING_TO_TARGET == self._phase:
                # stops = self._navigator.get_all_waypoints()
                self._navigator.reset_full()
                targetLoc = self._getTarget(state, state.get_self()['is_carrying'][0])['location']
                print(self._getTarget(state, state.get_self()['is_carrying'][0])['location'])
                # print(targetLoc)
                self._navigator.add_waypoint(targetLoc)
                # self._navigator.add_waypoints(stops)
                self._set_lazy(len(self._navigator.get_all_waypoints()))
                self._phase = Phase.BRING_TO_TARGET
            if Phase.BRING_TO_TARGET == self._phase:
                if not self._check_if_lazy():
                    self._state_tracker.update(state)
                    # Follow path to door
                    action = self._navigator.get_move_action(self._state_tracker)
                    if action != None:
                        return action, {}
                self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR
                target = self._getTarget(state, state.get_self()['is_carrying'][0])
                self._sendMessage('Dropped goal block ' + str(target['visualization']) + ' at location ' + str(
                    target['location']), agent_name, state)

                if self._target_missing_and_at_target_location(target['visualization'], target['location']):
                    key = self._get_missing_dict_key(target['visualization'])
                    print(key)
                    del self._missing[key]
                    self._current_target_block = str(int(self._current_target_block) + 1)

                # if target['visualization'] in self._missing and target['location'] == self._missing_locations[0]:
                #     self._missing.remove(target['visualization'])
                return DropObject.__name__, {'object_id': state.get_self()['is_carrying'][0]['obj_id']}

    def _check_if_lazy(self):
        if self._be_lazy:
            if self._be_lazy_after_moves == 0:
                self._be_lazy = False
                return True
            else:
                self._be_lazy_after_moves -= 1
                return False

    def _set_lazy(self, amountOfMoves):
        self._be_lazy = False
        self._be_lazy_after_moves = 0
        if random.random() > 0.5:
            self._be_lazy = True
            self._be_lazy_after_moves = random.randint(1, amountOfMoves)

    def _target_missing_and_at_target_location(self, target, target_location):
        for key in self._missing:
            if self._missing.get(key)['block'] == target:
                if self._missing.get(key)['location'] == target_location:
                    return True
                else:
                    return False
        return False

    def _get_missing_dict_key(self, target):
        for key in self._missing:
            if self._missing.get(key)['block'] == target:
                return key

    def _sendMessage(self, mssg, sender, state):
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
        dropped_messages = [msg.content for msg in self.received_messages
                            if Messages.DROPPED_GOAL_BLOCK1.value[0] in msg.content]
        for msg in dropped_messages:
            first = msg.split('block ')[1]
            block = first.split(' at')[0].replace("\'", '\"')
            location = msg.split('location ')[1]
            jsonblock = json.loads(block)
            if self._target_missing_and_at_target_location(jsonblock, location):
                self._missing.pop(self._get_missing_dict_key(jsonblock))
                self._current_target_block = str(int(self._current_target_block) + 1)

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

    def _getTargetBlocks(self, state):
        visualisations = copy.deepcopy([tile for tile in state.values()
                                        if 'class_inheritance' in tile
                                        and 'CollectableBlock' in tile['class_inheritance']
                                        and len(tile['carried_by']) == 0
                                        ])
        for i in range(len(visualisations)):
            visualisations[i]['visualization'].pop('opacity')
            visualisations[i]['visualization'].pop('depth')
            visualisations[i]['visualization'].pop('visualize_from_center')
        matching = [x for x in visualisations
                    if x['visualization'] == self._missing[self._current_target_block]['block']
                    ]
        return matching

    def _getTarget(self, state, block):
        target_blocks = copy.deepcopy([tile for tile in state.values()
                                       if 'class_inheritance' in tile
                                       and 'GhostBlock' in tile['class_inheritance']
                                       ])
        for i in range(len(target_blocks)):
            target_blocks[i]['visualization'].pop('opacity')
            target_blocks[i]['visualization'].pop('visualize_from_center')
            target_blocks[i]['visualization'].pop('depth')

        blockCopy = copy.deepcopy(block)['visualization']

        blockCopy.pop('opacity')
        blockCopy.pop('depth')
        blockCopy.pop('visualize_from_center')
        # print(blockCopy)
        # print(target_blocks)

        for tile in target_blocks:
            if tile['visualization'] == blockCopy:
                return tile

    def _getTargetFromVisualization(self, state, block):
        target_blocks = copy.deepcopy([tile for tile in state.values()
                                       if 'class_inheritance' in tile
                                       and 'GhostBlock' in tile['class_inheritance']
                                       ])
        for i in range(len(target_blocks)):
            target_blocks[i]['visualization'].pop('opacity')
            target_blocks[i]['visualization'].pop('visualize_from_center')
            target_blocks[i]['visualization'].pop('depth')

        block.pop('opacity')
        block.pop('depth')
        block.pop('visualize_from_center')

        for tile in target_blocks:
            if tile['visualization'] == block:
                # print('huh')
                return tile

