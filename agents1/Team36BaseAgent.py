import copy
import sys
from collections import defaultdict
from operator import le
from typing import final, List, Dict, Final
from functools import reduce
import json
from math import sqrt
import enum, random
from agents1.BW4TBaselineAgent import BaseLineAgent
from matrx.agents.agent_utils.state import State
from matrx.agents.agent_utils.navigator import Navigator
from matrx.agents.agent_utils.state_tracker import StateTracker
from matrx.actions.door_actions import OpenDoorAction
from matrx.actions.object_actions import GrabObject, DropObject
from matrx.messages.message import Message


class Phase(enum.Enum):
    PLAN_NEXT_ACTION = 0,
    PLAN_PATH_TO_CLOSED_DOOR = 1,
    PLAN_PATH_TO_ROOM = 2,
    PLAN_PATH_TO_BLOCK = 3,
    PLAN_PATH_TO_GOAL = 4,
    PLAN_SEARCH_ROOM = 5
    FOLLOW_PATH_TO_CLOSED_DOOR = 6,
    FOLLOW_PATH_TO_ROOM = 7,
    FOLLOW_PATH_TO_BLOCK = 8,
    FOLLOW_PATH_TO_GOAL = 9,
    OPEN_DOOR = 10,
    PICKUP_BLOCK = 11,
    DROP_BLOCK = 12,
    SEARCH_ROOM = 13

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

class Status(enum.Enum):
    MOVING_TO_ROOM = 0,
    MOVING_TO_GOAL = 1,


# TODO: message sending
# TODO: message handling
# TODO: keep track of status (carrying, color, room, etc) of team members
class BaseAgent(BaseLineAgent):

    def __init__(self, settings: Dict[str, object]):
        super().__init__(settings)
        self._unsearched_rooms = None
        self._rooms_searched_by_teammembers = []
        self._missing_blocks = {}
        self._found_blocks = defaultdict(set)
        self._current_room = None
        self._door = None
        self._phase = Phase.PLAN_NEXT_ACTION
        self.gf_start = None
        self._teammember_states = {}  # states of teammembers will be added in this dict with the key being the teammembers ID
        self._current_state = {'type': None}
        self._index = -1
        self._carrying_capacity = 1
        self._current_target_block = '0'

    def initialize(self):
        super().initialize()
        self._state_tracker = StateTracker(agent_id=self.agent_id)
        self._navigator = Navigator(agent_id=self.agent_id,
                                    action_set=self.action_set, algorithm=Navigator.A_STAR_ALGORITHM)


    def _handleMessages(self, messages):
        for member in self._teamMembers:
            if member not in messages:
                continue
            for msg in messages[member]:
                if 'Moving to ' in msg and 'door' not in msg:
                    room_id = msg.replace("Moving to ", "", 1)
                    self._rooms_searched_by_teammembers.append(room_id)
                    self._teammember_states[member]['state'] = {'type': Status.MOVING_TO_ROOM, 'room': room_id}

    def filter_bw4t_observations(self, state):
        return state

    def decide_on_bw4t_action(self, state: State):
        agent_name = state[self.agent_id]['obj_id']
        self._you = state[self.agent_id]
        # Add team members
        for member in state['World']['team_members']:
            index = state['World']['team_members'].index(member)
            if self._index == -1 and member == agent_name:
                self._index = index
            if member != agent_name and member not in self._teammember_states:
                self._teammember_states[member] = {}
                self._teammember_states[member]['index'] = index
                self._teammember_states[member]['state'] = {'type': None}
            if member != agent_name and member not in self._teamMembers:
                self._teamMembers.append(member)
                # add required blocks
        if not self._missing_blocks:
            self.init_missing_blocks(state)

        # initialize unsearched rooms list
        if self._unsearched_rooms == None:
            self._unsearched_rooms = state.get_all_room_names()
        # Process messages from team members
        receivedMessages = self._processMessages(self._teamMembers)
        self._handleMessages(receivedMessages)
        # Update trust beliefs for team members
        self._trustBlief(self._teamMembers, receivedMessages)

        closedDoors = [door for door in state.values()
                       if 'class_inheritance' in door and 'Door' in door['class_inheritance'] and not door['is_open']]

        # if a teammate with a higher 'dominance', aka index, is searching the same room as us move to another possible room
        for member_states in self._teammember_states.values():
            # print("member index: {}, my index: {}".format(member_states['index'], self._index))
            # print(str(member_states['state']))
            # print(str(self._current_state))
            if (member_states['index'] < self._index
                    and member_states['state']['type'] == Status.MOVING_TO_ROOM
                    and self._current_state == member_states['state']):
                self._phase = Phase.PLAN_NEXT_ACTION
                print("NEED TO PICK ANOTHER ROOM")
                break

        while True:
            if Phase.PLAN_NEXT_ACTION == self._phase:
                self.plan_next_action(state)

            if Phase.PLAN_PATH_TO_ROOM == self._phase:
                self.plan_path_to_room(agent_name, state)

            if Phase.PLAN_PATH_TO_CLOSED_DOOR == self._phase:
                self.plan_path_to_closed_door(agent_name, closedDoors, state)

            if Phase.PLAN_PATH_TO_BLOCK == self._phase:
                self.plan_path_to_block(state)

            if Phase.PLAN_PATH_TO_GOAL == self._phase:
                self.plan_path_to_goal(state)

            if Phase.FOLLOW_PATH_TO_ROOM == self._phase:
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    return action, {}
                self._phase = Phase.PLAN_SEARCH_ROOM

            if Phase.FOLLOW_PATH_TO_BLOCK == self._phase:
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    return action, {}
                else:
                    self._phase = Phase.PICKUP_BLOCK

            if Phase.FOLLOW_PATH_TO_GOAL == self._phase:
                target = self._getTarget(state, state.get_self()['is_carrying'][0])
                # if currently carried block already found, drop it and start searching for the next one
                if str(target['visualization']) != str(self._missing_blocks[self._current_target_block]["block"]):
                    self._phase = Phase.DROP_BLOCK
                    continue
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    return action, {}
                else:
                    self._phase = Phase.DROP_BLOCK

            if Phase.FOLLOW_PATH_TO_CLOSED_DOOR == self._phase:
                self._state_tracker.update(state)
                # Follow path to door
                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    return action, {}
                self._phase = Phase.OPEN_DOOR

            if Phase.OPEN_DOOR == self._phase:
                return self.open_door(agent_name)

            if Phase.PLAN_SEARCH_ROOM == self._phase:
                self.plan_search_room(agent_name, state)

            if Phase.SEARCH_ROOM == self._phase:
                self._state_tracker.update(state)
                matching = self._getTargetBlocks(state)
                for block in matching:
                    self._sendMessage(
                        'Found goal block ' + str(block['visualization']) + ' at location ' + str(block['location']),
                        agent_name)
                    if str(block['visualization']) in self._found_blocks.keys():
                        locations = self._found_blocks[str(block['visualization'])]
                        locations.add(block['location'])
                        self._found_blocks[str(block['visualization'])] |= locations
                    else:
                        self._found_blocks[str(block['visualization'])] |= {block['location']}

                    if str(self._missing_blocks[self._current_target_block]) == str(block['visualization']):
                        self._block = block
                        self._phase = Phase.PICKUP_BLOCK

                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    return action, {}
                # room is completely searched
                self._unsearched_rooms.remove(self._current_room['room_name'])
                self._phase = Phase.PLAN_NEXT_ACTION

            if Phase.PICKUP_BLOCK == self._phase:
                return self.pickup_block(agent_name, state)

            if Phase.DROP_BLOCK == self._phase:
                return self.drop_block(agent_name, state)

    def plan_next_action(self, state):
        self._navigator.reset_full()
        moving_to_target = False
        if len(state[self.agent_id]['is_carrying']) < self._carrying_capacity and len(state[self.agent_id]['is_carrying']) < len(
                self._missing_blocks):
            for block in self._found_blocks.keys():
                if str(self._missing_blocks[self._current_target_block]['block']) == str(block):
                    self._phase = Phase.PLAN_PATH_TO_BLOCK
                    moving_to_target = True
        elif len(state[self.agent_id]['is_carrying']) > 0:
            self._phase = Phase.PLAN_PATH_TO_GOAL
            moving_to_target = True
        if not moving_to_target:
            self._phase = Phase.PLAN_PATH_TO_ROOM

    def plan_path_to_room(self, agent_name, state):
        # we will find and create a path to a room that is not yet searched
        # get a list of all rooms that are not yet traversed/searched (completely)
        possible_rooms = [room for room in state.values()
                          if 'class_inheritance' in room and 'Door' in room['class_inheritance'] and room[
                              'room_name'] in self._unsearched_rooms]
        # remove rooms that are already being searched by teammembers from possible roomslass_inheritance'] and room['name'] == member['state']['room']])
        possible_rooms = [room for room in possible_rooms if
                          room['room_name'] not in self._rooms_searched_by_teammembers]
        if len(possible_rooms) == 0:
            # some rooms where not completely searched apparently, so restart searching all rooms
            possible_rooms = [room for room in state.values()
                              if 'class_inheritance' in room and 'Door' in room['class_inheritance']]
            self._unsearched_rooms = state.get_all_room_names()
        # find closest room
        move_to_room = min(possible_rooms, key=lambda r: self.dist(self._you, r, state=state))
        self._sendMessage('Moving to {}'.format(move_to_room['room_name']), agent_name)
        self._current_state = {'type': Status.MOVING_TO_ROOM, 'room': move_to_room['room_name']}
        self._current_room = move_to_room
        if not move_to_room['is_open']:
            self._door = move_to_room
            self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR
        else:
            self._navigator.add_waypoints([move_to_room['location']])
            self._phase = Phase.FOLLOW_PATH_TO_ROOM

    def plan_path_to_closed_door(self, agent_name, closedDoors, state):
        if self._door == None:
            # Pick door based on the closest manhattan distance
            self._door = min(closedDoors, key=lambda d: self.dist(self._you, d, state=state))
        doorLoc = self._door['location']
        # Location in front of door is south from door
        doorLoc = doorLoc[0], doorLoc[1] + 1
        # Send message of current action
        self._sendMessage('Moving to door of ' + self._door['room_name'], agent_name)
        self._navigator.add_waypoints([doorLoc])
        self._phase = Phase.FOLLOW_PATH_TO_CLOSED_DOOR

    def plan_path_to_block(self, state):
        possible_target = self.get_closest_possible_target(state)
        if possible_target == None:
            self._phase = Phase.PLAN_PATH_TO_ROOM
        else:
            self._navigator.reset_full()
            self._navigator.add_waypoints([possible_target[1]])
            self._block = {'visualization': possible_target[0], 'location': possible_target[1]}
            self._phase = Phase.FOLLOW_PATH_TO_BLOCK

    def plan_path_to_goal(self, state):
        if len(state.get_self()['is_carrying']) == 0:
            self._phase = Phase.PLAN_NEXT_ACTION
        else:
            targetLoc = self.get_target(state, state.get_self()['is_carrying'][0])['location']
            self._navigator.reset_full()
            self._navigator.add_waypoint(targetLoc)
            self._phase = Phase.FOLLOW_PATH_TO_GOAL

    def open_door(self, agent_name):
        self._phase = Phase.PLAN_SEARCH_ROOM
        # Open door
        self._sendMessage('Opening door of {}'.format(self._current_room['room_name']), agent_name)
        action = OpenDoorAction.__name__, {'object_id': self._door['obj_id']}
        self._door = None
        return action

    def plan_search_room(self, agent_name, state):
        self._navigator.reset_full()
        room_tiles = [tile for tile in state.values()
                      if 'room_name' in tile and self._current_room['room_name'] is tile['room_name']
                      and 'AreaTile' in tile['class_inheritance'] and tile['is_traversable']]
        for tile in room_tiles:
            self._navigator.add_waypoints([tile['location']])
        self._sendMessage('Searching through {}'.format(self._current_room['room_name']), agent_name)
        self._phase = Phase.SEARCH_ROOM

    def pickup_block(self, agent_name, state):
        print("picking up block: {}".format(self._block['visualization']))
        matching = self._getTargetBlocks(state)
        for block in matching:
            if block['visualization'] == self._missing_blocks[self._current_target_block]['block']:
                self._sendMessage(
                    'Picking up goal block ' + str(block['visualization']) + ' at location ' + str(
                        block['location']), agent_name)
                self._phase = Phase.PLAN_PATH_TO_GOAL
                if str(block['visualization']) in self._found_blocks.keys() and self._found_blocks[
                    str(block['visualization'])] == str(block['location']):
                    del self._found_blocks[str(block['visualization'])]
                return GrabObject.__name__, {'object_id': block['obj_id']}
        self._phase = Phase.PLAN_NEXT_ACTION

    def drop_block(self, agent_name, state):
        target = self._getTarget(state, state.get_self()['is_carrying'][0])
        self._phase = Phase.PLAN_NEXT_ACTION
        self._sendMessage('Dropped goal block ' + str(target['visualization']) + ' at location ' + str(
            state.get_self()['location']), agent_name)
        if self.target_missing_and_at_goal(target['visualization'], state.get_self()['location']):
            del self._missing_blocks[self._current_target_block]
            self._current_target_block = str(int(self._current_target_block) + 1)
        else:
            self._found_blocks[str(target['visualization'])] = state.get_self()['location']
        return DropObject.__name__, {'object_id': state.get_self()['is_carrying'][0]['obj_id']}

    def target_missing_and_at_goal(self, block, location):
        return str(block) == str(self._missing_blocks[self._current_target_block]['block']) and str(location) == str(
            self._missing_blocks[self._current_target_block]['location'])

    def init_missing_blocks(self, state):
        blocks = copy.deepcopy([tile['visualization'] for tile in state.values()
                                if 'class_inheritance' in tile
                                and 'GhostBlock' in tile['class_inheritance']
                                ])
        for i in range(len(blocks)):
            location = (self._get_goal_from_visualization(state, blocks[i])['location'])
            self._missing_blocks[str(i)] = {'block': blocks[i], 'location': location}

    def get_target(self, state, block):
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
            for block in self._missing_blocks.values():
                if str(x['visualization']) == str(block['block']):
                    matching.append(x)

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

        for tile in target_blocks:
            if tile['visualization'] == blockCopy:
                return tile

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
                    #self._team_member_moving_to[member] = room_name

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
            if str(block) not in self._found_blocks.keys():
                self._found_blocks[str(block)] = {location}
            else:
                other = self._found_blocks[str(block)]
                other.add(location)
                self._found_blocks[str(block)] |= other


        # picking up goal block [block_vis] at location [location]
        picked_up_messages = [msg.content for msg in self.received_messages
                              if Messages.PICKING_UP_GOAL_BLOCK1.value[0] in msg.content]
        for msg in picked_up_messages:
            first = msg.split('block ')[1]
            block = first.split(' at')[0].replace("\'", '\"')
            location = eval(first.split('location ')[1])
            if str(block) in self._found_blocks.keys() and location in self._found_blocks[str(block)]:
                locations = self._found_blocks[str(block)]
                locations.remove(location)
                if locations:
                    self._found_blocks[str(block)] |= locations
                else:
                    self._found_blocks.pop(str(block))
            # self._picked_up.append(block)

        # dropped goal block [block_vis] at location [location]
        dropped_messages = [msg.content for msg in self.received_messages
                            if Messages.DROPPED_GOAL_BLOCK1.value[0] in msg.content]
        for msg in dropped_messages:
            first = msg.split('block ')[1]
            block = first.split(' at')[0].replace("\'", '\"')
            location = eval(msg.split('location ')[1])
            jsonblock = json.loads(block)
            if self.target_missing_and_at_goal(jsonblock, location):
                self._missing_blocks.pop(self._current_target_block)
                self._current_target_block = str(int(self._current_target_block) + 1)
            else:
                if not self._found_blocks[str(block)]:
                    self._found_blocks[str(block)] = {location}
                else:
                    other = self._found_blocks[str(block)]
                    self._found_blocks[str(block)] = other.add(location)

        receivedMessages = {}
        for member in teamMembers:
            receivedMessages[member] = []
        for mssg in self.received_messages:
            for member in teamMembers:
                if mssg.from_id == member:
                    receivedMessages[member].append(mssg.content)

        self.received_messages.clear()
        return receivedMessages


    def create_gf_field(self, state):
        # TODO take moved pieces into account
        if self.gf_start != None and state.get_self()['location'][0] == self.gf_start[0] and \
                state.get_self()['location'][1] == self.gf_start[1]:
            return

        print("creating grassfire algo field")
        map = state.get_traverse_map()
        doors = [door['location'] for door in state.values() if
                 'class_inheritance' in door and 'Door' in door['class_inheritance']]
        start = state.get_self()['location']
        width, length = state.get_world_info()['grid_shape']
        self.gf_field = []
        for x in range(width):
            self.gf_field.append([])
            for y in range(length):
                val = -1
                if not map[(x, y)]:
                    val = -2
                self.gf_field[x].append(
                    val)  # this grassfire map will have -1 if not yet traversed, and -2 if it can't be traversed

        # for this we want doors to always be traversable
        for door in doors:
            self.gf_field[door[0]][door[1]] = -1

        q = []
        new_q = []
        level = 0
        q.append(start)

        while len(q) > 0:
            l = q.pop(0)
            self.gf_field[l[0]][l[1]] = level

            # add new frontier positions
            for dx in [-1, 0, 1]:
                x = l[0] + dx
                for dy in [-1, 0, 1]:
                    y = l[1] + dy
                    if dx == 0 and dy == 0:  # dont do same spot
                        pass
                    elif dx != 0 and dy != 0:  # dont do diagonals
                        pass
                    elif x < 0 or y < 0 or x >= width or y >= length:  # dont do outside of map
                        pass
                    elif self.gf_field[x][y] == -1:  # or gf_field[x][y] > level+1:  # do unvisited or
                        t = (x, y)
                        new_q.append(t)

            # if the current frontier is empty replace it with the new frontier and increase the distance (level)
            if len(q) == 0:
                q = new_q
                new_q = []
                level += 1

        self.gf_start = (state.get_self()['location'][0], state.get_self()['location'][1])

    def dist(self, start, target, state=None):
        if state == None:
            return sqrt((start['location'][0] - target['location'][0]) ** 2 + (
                        start['location'][1] - target['location'][1]) ** 2)
        else:
            # calculate grassfire heuristic, will only calculate the field if necessary, aka pieces/you moved
            self.create_gf_field(state)
            return self.gf_field[target['location'][0]][target['location'][1]]

    def get_closest_possible_target(self, state):
        min_dist = sys.maxsize
        target_location = None
        for block in self._found_blocks.keys():
            if str(self._missing_blocks[self._current_target_block]['block']) == str(block):
                locations = self._found_blocks[block]
                for location in locations:
                    dist = self.dist(self._you, {'location': location}, state)
                    if dist < min_dist:
                        target_location = location
                return block, target_location
        return None

    def _get_goal_from_visualization(self, state, block):
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