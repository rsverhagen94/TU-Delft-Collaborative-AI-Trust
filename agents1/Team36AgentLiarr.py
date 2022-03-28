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
    PLAN_BRING_TO_TARGET = 8,
    PLAN_PATH_TO_TARGET_BLOCK_IF_FOUND = 9,
    MOVE_TO_TARGET_BLOCK = 10


class Messages(enum.Enum):
    MOVING_TO_ROOM = "Moving to ",
    OPENING_DOOR = "Opening door of ",
    SEARCHING_THROUGH = "Searching through ",
    FOUND_GOAL_BLOCK1 = "Found goal block ",
    FOUND_GOAL_BLOCK2 = " at location ",
    PICKING_UP_GOAL_BLOCK1 = "Picking up goal block ",
    PICKING_UP_GOAL_BLOCK2 = " at location ",
    DROPPED_GOAL_BLOCK1 = "Dropped goal block ",
    DROPPED_GOAL_BLOCK2 = " at drop location ",


class Liar(BW4TBrain):

    def __init__(self, settings: Dict[str, object]):
        super().__init__(settings)
        self._dropped_blocks = {}
        self._found_blocks = {}
        self._current_target_block = '0'
        self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR
        self._teamMembers = []
        self._missing = {}
        self._be_lazy = False
        self._be_lazy_after_moves = 0
        self._team_member_moving_to = {}
        self._opened_doors = []
        self._searched_rooms = []
        self._picked_up = []

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
        if not self._missing:
            blocks = copy.deepcopy([tile['visualization'] for tile in state.values()
                                    if 'class_inheritance' in tile
                                    and 'GhostBlock' in tile['class_inheritance']
                                    ])
            for i in range(len(blocks)):
                location = (self._getTargetFromVisualization(state, blocks[i])['location'])
                self._missing[str(i)] = {'block': blocks[i], 'location': location}

        # Process messages from team members
        receivedMessages = self._processMessages(self._teamMembers)
        # Update trust beliefs for team members
        self._trustBlief(self._teamMembers, receivedMessages)

        while True:
            if Phase.PLAN_PATH_TO_TARGET_BLOCK_IF_FOUND == self._phase:
                if str(self._missing[self._current_target_block]['block']) in self._found_blocks.keys():
                    self._navigator.reset_full()
                    self._navigator.add_waypoint(
                    self._found_blocks[str(self._missing[self._current_target_block]['block'])])
                    self._phase = Phase.MOVE_TO_TARGET_BLOCK
                else:
                    self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR

            if Phase.MOVE_TO_TARGET_BLOCK == self._phase:
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    return action, {}
                matching = self._getTargetBlocks(state)
                for block in matching:
                    if block['visualization'] == self._missing[self._current_target_block]['block']:
                        self._sendMessage(
                            'Picking up goal block ' + str(block['visualization']) + ' at location ' + str(
                                block['location']), agent_name, state)
                        self._phase = Phase.PLAN_BRING_TO_TARGET
                        if str(block['visualization']) in self._found_blocks.keys() and self._found_blocks[
                            str(block['visualization'])] == str(block['location']):
                            del self._found_blocks[str(block['visualization'])]
                        return GrabObject.__name__, {'object_id': block['obj_id']}
                self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR

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
                self._phase = Phase.FOLLOW_PATH_TO_CLOSED_DOOR

            if Phase.FOLLOW_PATH_TO_CLOSED_DOOR == self._phase:
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    return action, {}
                if str(self._missing[self._current_target_block]['block']) in self._found_blocks.keys():
                    self._phase = Phase.PLAN_PATH_TO_TARGET_BLOCK_IF_FOUND
                    self._navigator.reset_full()
                    continue
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
                self._phase = Phase.PLAN_MOVE_IN_ROOM

            if Phase.PLAN_MOVE_IN_ROOM == self._phase:
                self._state_tracker.update(state)
                self._navigator.reset_full()
                if len(self._room_tiles) > 0:
                    tileLoc = self._room_tiles.pop()['location']
                    self._navigator.add_waypoints([tileLoc])
                    self._phase = Phase.MOVE_IN_ROOM
                else:
                    self._phase = Phase.PLAN_PATH_TO_TARGET_BLOCK_IF_FOUND

            if Phase.MOVE_IN_ROOM == self._phase:
                self._state_tracker.update(state)
                matching = self._getTargetBlocks(state)
                for block in matching:
                    self._sendMessage(
                        'Found goal block ' + str(block['visualization']) + ' at location ' + str(block['location']),
                        agent_name, state)
                    self._found_blocks[str(block['visualization'])] = block['location']
                for block in matching:
                    if block['visualization'] == self._missing[self._current_target_block]['block']:
                        self._sendMessage(
                            'Picking up goal block ' + str(block['visualization']) + ' at location ' + str(
                                block['location']), agent_name, state)
                        self._phase = Phase.PLAN_BRING_TO_TARGET
                        if str(block['visualization']) in self._found_blocks.keys() and self._found_blocks[
                            str(block['visualization'])] == str(block['location']):
                            del self._found_blocks[str(block['visualization'])]
                        return GrabObject.__name__, {'object_id': block['obj_id']}
                self._state_tracker.update(state)
                # Follow path to tile
                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    return action, {}
                self._phase = Phase.PLAN_MOVE_IN_ROOM

            if Phase.PLAN_BRING_TO_TARGET == self._phase:
                if len(state.get_self()['is_carrying']) == 0:
                    self._phase = Phase.MOVE_IN_ROOM
                else:
                    targetLoc = self._getTarget(state, state.get_self()['is_carrying'][0])['location']
                    self._navigator.reset_full()
                    self._navigator.add_waypoint(targetLoc)
                    self._phase = Phase.BRING_TO_TARGET

            if Phase.BRING_TO_TARGET == self._phase:
                target = self._getTarget(state, state.get_self()['is_carrying'][0])
                #if currently carried block already found, drop it and start searching for the next one
                if str(target['visualization']) != str(self._missing[self._current_target_block]["block"]):
                    self._found_blocks[str(target['visualization'])] = state.get_self()['location']
                    self._phase = Phase.PLAN_PATH_TO_TARGET_BLOCK_IF_FOUND
                    self._sendMessage('Dropped goal block ' + str(target['visualization']) + ' at location ' + str(
                        state.get_self()['location']), agent_name, state)
                    return DropObject.__name__, {'object_id': state.get_self()['is_carrying'][0]['obj_id']}
                self._state_tracker.update(state)
                # Follow path to door
                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    return action, {}
                self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR
                self._sendMessage('Dropped goal block ' + str(target['visualization']) + ' at location ' + str(
                    state.get_self()['location']), agent_name, state)
                if self._target_missing_and_at_target_location(target['visualization'], state.get_self()['location']):
                    del self._missing[self._current_target_block]
                    self._current_target_block = str(int(self._current_target_block) + 1)
                else:
                    self._found_blocks[str(target['visualization'])] = state.get_self()['location']
                return DropObject.__name__, {'object_id': state.get_self()['is_carrying'][0]['obj_id']}

    def _target_missing_and_at_target_location(self, target, target_location):
        if self._missing.get(self._current_target_block)['block'] == target:
            if str(self._missing.get(self._current_target_block)['location']) == str(target_location):
                return True
        return False

    def _sendMessage(self, mssg, sender, state):
        '''
        Enable sending messages in one line of code
        '''
        msg = Message(content=mssg, from_id=sender)
        doors = [door['room_name'] for door in state.values()
                 if 'class_inheritance' in door
                 and 'Door' in door['class_inheritance']]
        blocks = self._getTargets(state)
        randommessage = random.choice([Messages.MOVING_TO_ROOM,
                                       Messages.OPENING_DOOR,
                                       Messages.SEARCHING_THROUGH,
                                       Messages.FOUND_GOAL_BLOCK1,
                                       Messages.PICKING_UP_GOAL_BLOCK2,
                                       Messages.DROPPED_GOAL_BLOCK1])
        message = ''
        if Messages.MOVING_TO_ROOM == randommessage:
            message = Messages.MOVING_TO_ROOM.value[0] + random.choice(doors)
        elif Messages.OPENING_DOOR == randommessage:
            message = Messages.OPENING_DOOR.value[0] + random.choice(doors)
        elif Messages.SEARCHING_THROUGH == randommessage:
            message = Messages.SEARCHING_THROUGH.value[0] + random.choice(doors)
        elif Messages.FOUND_GOAL_BLOCK1 == randommessage:
            block = random.choice(blocks)
            message = Messages.FOUND_GOAL_BLOCK1.value[0] + repr(block['visualization']) + \
                      Messages.FOUND_GOAL_BLOCK2.value[0] + repr(block['location'])
        elif Messages.PICKING_UP_GOAL_BLOCK1 == randommessage:
            block = random.choice(blocks)
            message = Messages.PICKING_UP_GOAL_BLOCK1.value[0] + repr(block['visualization']) + \
                      Messages.PICKING_UP_GOAL_BLOCK2.value[0] + repr(block['location'])
        elif Messages.DROPPED_GOAL_BLOCK1 == randommessage:
            block = random.choice(blocks)
            message = Messages.DROPPED_GOAL_BLOCK1.value[0] + repr(block['visualization']) + \
                      Messages.DROPPED_GOAL_BLOCK2.value[0] + repr(block['location'])

        if random.random() > 0.2:
            msg = Message(content=message, from_id=sender)
        if msg.content not in self.received_messages:
            self.send_message(msg)

    def _processMessages(self, teamMembers):
        '''
        Process incoming messages and create a dictionary with received messages from each team member.
        '''

        # moving to [room name]
        moving_to_messages = [msg for msg in self.received_messages
                              if Messages.MOVING_TO_ROOM.value[0] in msg.content]
        for msg in moving_to_messages:
            for member in teamMembers:
                if msg.from_id == member:
                    room_name = msg.content.split('to ')
                    self._team_member_moving_to[member] = room_name

        # opening door of [room name]
        opening_messages = [msg.content for msg in self.received_messages
                            if Messages.OPENING_DOOR.value[0] in msg.content]
        for msg in opening_messages:
            room_name = msg.split('of ')
            # self._opened_doors.append(room_name)

        # searching through [room name]
        searching_messages = [msg.content for msg in self.received_messages
                              if Messages.SEARCHING_THROUGH.value[0] in msg.content]
        for msg in searching_messages:
            room_name = msg.split('through ')
            # self._searched_rooms.append(room_name)

        # found goal block [block_vis] at location [location]
        found_messages = [msg.content for msg in self.received_messages
                          if Messages.FOUND_GOAL_BLOCK1.value[0] in msg.content]
        for msg in found_messages:
            first = msg.split('block ')[1]
            block = first.split(' at')[0].replace("\'", '\"')
            location = eval(first.split('location ')[1])
            self._found_blocks[str(block)] = location

        # picking up goal block [block_vis] at location [location]
        picked_up_messages = [msg.content for msg in self.received_messages
                              if Messages.PICKING_UP_GOAL_BLOCK1.value[0] in msg.content]
        for msg in picked_up_messages:
            first = msg.split('block ')[1]
            block = first.split(' at')[0].replace("\'", '\"')
            if str(block) in self._found_blocks.keys():
                del self._found_blocks[str(block)]
            # self._picked_up.append(block)

        # dropped goal block [block_vis] at location [location]
        dropped_messages = [msg for msg in self.received_messages
                            if Messages.DROPPED_GOAL_BLOCK1.value[0] in msg.content]
        for msg in dropped_messages:
            text = msg.content
            print(text)
            first = text.split('block ')[1]
            block = first.split(' at')[0].replace("\'", '\"')
            location = eval(text.split('location ')[1])
            jsonblock = json.loads(block)
            if self._target_missing_and_at_target_location(jsonblock, location):
                del self._missing[self._current_target_block]
                self._current_target_block = str(int(self._current_target_block) + 1)
            else:
                self._found_blocks[str(block)] = location

        receivedMessages = {}
        for member in teamMembers:
            receivedMessages[member] = []
        for mssg in self.received_messages:
            for member in teamMembers:
                if mssg.from_id == member:
                    receivedMessages[member].append(mssg.content)

        self.received_messages.clear()
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
        matching = []
        for x in visualisations:
            for block in self._missing.values():
                if x['visualization'] == block['block']:
                    matching.append(x)

        return matching

    def _getTargets(self, state):
        target_blocks = copy.deepcopy([tile for tile in state.values()
                                       if 'class_inheritance' in tile
                                       and 'GhostBlock' in tile['class_inheritance']
                                       ])
        for i in range(len(target_blocks)):
            target_blocks[i]['visualization'].pop('opacity')
            target_blocks[i]['visualization'].pop('visualize_from_center')
            target_blocks[i]['visualization'].pop('depth')

        return target_blocks

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
                return tile