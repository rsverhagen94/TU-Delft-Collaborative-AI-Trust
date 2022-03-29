import copy
import re
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

class State(enum.Enum):
    MOVING_TO_ROOM = 0,
    MOVING_TO_GOAL = 1,
    PICKING_UP_BLOCK = 2,


# TODO: fix missing information for block visualization (like for colour blind) or additional (useless) information, cuz who knows what whacky stuff other agents do
class BaseAgent(BaseLineAgent):

    def __init__(self, settings: Dict[str, object]):
        super().__init__(settings)

        self._current_room = None
        self._door = None
        self._carrying_capacity = 1
        self.gf_start = None
        self._current_state = {'type': None}
        self._world_state = {
            'found_blocks': [],  # list of blocks, contains {'location','visualization','by''}
            'teammembers': {},  # dict of teammembers accessible by their id, contains {'state','carrying','index'}
            'goals': [],  # list of goals, contains {'location','visualization','index','satisfied','by','checked'}
            'searched_rooms': [],  # list of rooms, contains {'room_id','by'}
            'opened_doors': [],  # list of doors, contains {'room_id','by'}
            'agent_index': -1,  # index/dominance of this agent
        }

    def initialize(self):
        super().initialize()
        self._state_tracker = StateTracker(agent_id=self.agent_id)
        self._navigator = Navigator(agent_id=self.agent_id,
                                    action_set=self.action_set, algorithm=Navigator.A_STAR_ALGORITHM)

        self._current_room = None
        self._door = None
        self._phase = Phase.PLAN_NEXT_ACTION
        self.gf_start = None
        self._current_state = {'type': None}
        self._carrying_capacity = 1
        self._current_target_block = '0'
        self._beliefs = {}

        self._world_state = {
            'found_blocks': [],     # list of blocks, contains {'location','visualization','by'(,'obj_id')}
            'teammembers': {},      # dict of teammembers accessible by their id, contains {'state','carrying','index'}
            'goals': [],            # list of goals, contains {'location','visualization','index','satisfied','by','checked'}
            'searched_rooms': [],   # list of rooms, contains {'room_id','by'}
            'opened_doors': [],     # list of doors, contains {'room_id','by'}
            'agent_index': -1,      # index/dominance of this agent
        }

    def _processMessages(self, teamMembers):
        messages = super()._processMessages(teamMembers)
        self.received_messages.clear()
        current_world_state = self._world_state

        for member in messages.keys():
            if member not in current_world_state['teammembers']:
                # if this member is not in the world state it does not exist
                continue
            msgs = messages[member]
            for msg in msgs:
                if 'Moving to ' in msg:
                    room_id = msg.split(' ')[-1]
                    current_world_state['teammembers'][member]['state'] = {'type': State.MOVING_TO_ROOM, 'room_id': room_id}
                    current_world_state['searched_rooms'].append({'room_id': room_id, 'by': member})

                elif 'Opening door of ' in msg:
                    room_id = msg.split(' ')[-1]
                    current_world_state['opened_doors'].append({'room_id': room_id, 'by': member})

                elif 'Found goal block ' in msg:
                    vis_and_loc = msg.replace('Found goal block ', '', 1).split(' at location ')
                    block_vis = json.loads(vis_and_loc[0])
                    block_loc = eval(vis_and_loc[1])
                    block = {'visualization': block_vis, 'location': block_loc, 'by': member}
                    exists = False
                    for b in current_world_state['found_blocks']:
                        if b['visualization'] == block_vis and b['location'] == block_loc:
                            exists = True
                            break
                    if not exists:
                        current_world_state['found_blocks'].append(block)

                elif 'Picking up goal block ' in msg:
                    vis_and_loc = msg.replace('Picking up goal block ', '', 1).split(' at location ')
                    block_vis = json.loads(vis_and_loc[0])
                    block_loc = eval(vis_and_loc[1])
                    # remove picked up block from world state
                    for block in current_world_state['found_blocks']:
                        if block_vis == block['visualization'] and block_loc == block['location']:
                            current_world_state['found_blocks'].remove(block)
                            break
                    # add info to teammembers that a member is carrying a block
                    current_world_state['teammembers'][member]['carrying'].append(block_vis)

                elif 'Dropped goal block ' in msg:
                    vis_and_loc = msg.replace('Dropped goal block ', '', 1).split(' at drop location ')
                    block_vis = json.loads(vis_and_loc[0])
                    block_loc = eval(vis_and_loc[1])
                    current_world_state['found_blocks'].append(
                        {'visualization': block_vis, 'location': block_loc, 'by': member})
                    # remove info from teammembers that a member is carrying a block
                    try:
                        current_world_state['teammembers'][member]['carrying'].remove(block_vis)
                    except ValueError:
                        # member was not carrying block they said they dropped, thus is probably lying
                        self._decreaseBelief("trust", member, 0.3)
                    # test if location is supposedly goal, if so set goal to be SAT
                    for goal in current_world_state['goals']:
                        if block_loc == goal['location'] and block_vis == goal['visualization']:
                            goal['satisfied'] = True
                            goal['by'] = member

        return messages

    def _trustBelief(self, member, received):
        # You can change the default value to your preference
        default = 0.5
        for member in received.keys():
            for message in received[member]:
                if 'Dropped' in message:
                    vis_and_loc = message.replace('Dropped goal block ', '', 1).split(' at drop location ')
                    for block in self._world_state['goals']:
                        if block['visualization'] == vis_and_loc[0] and not block['satisfied']:
                            self._decreaseBelief("willingness", member, 0.1)

                if 'Found' in message and 'colour' not in message:
                    #update capability (agent does not provide all information)
                    self._decreaseBelief("capability", member, 0.1)

    def _increaseBelief(self, type, member, amount):
        if type == "willingness":
            self._beliefs[member]['willingness'] += amount
        elif type ==  "trust":
            self._beliefs[member]['trust'] += amount
        elif type == "competence":
            self._beliefs[member]['competence'] += amount

    def _decreaseBelief(self, type, member, amount):
        if type == "willingness":
            self._beliefs[member]['willingness'] -= amount
        elif type ==  "trust":
            self._beliefs[member]['trust'] -= amount
        elif type == "competence":
            self._beliefs[member]['competence'] -= amount

    def _handleMessages(self, state):
        # if a goal has been satisfied and you are carrying a block for that goal, drop the block
        missing_goals = self.get_missing_goals()
        is_goal = False
        for goal in missing_goals:
            if len(state.get_self()['is_carrying']) == 0:
                return
            carrying_vis = state.get_self()['is_carrying'][0]['visualization']
            if (carrying_vis['size'] == goal['visualization']['size']
                    and carrying_vis['shape'] == goal['visualization']['shape']
                    and carrying_vis['colour'] == goal['visualization']['colour']):
                is_goal = True

        if not is_goal:
            self._phase = Phase.DROP_BLOCK
        
    def _processObservations(self, state):
        observations = {
            'blocks': [],
            'teammembers': [],
            'doors': []
        }
        # blocks:
        blocks = copy.deepcopy([block for block in state.values()
                    if 'class_inheritance' in block and 'CollectableBlock' in block['class_inheritance'] and block['is_collectable']])
        for block in blocks:
            block_vis = block['visualization']
            block_vis.pop('opacity')
            block_vis.pop('visualize_from_center')
            block_vis.pop('depth')
            for goal in self._world_state['goals']:
                if (block_vis == goal['visualization']):
                    observations['blocks'].append({'obj_id': block['obj_id'], 'visualization': block_vis, 'location': block['location']})
                    break
        
        # teammembers:
        # doors:
            
        return observations
    
    def _handleObservations(self, observations):
        for block in observations['blocks']:
            self._sendMessage('Found goal block {} at location {}'.format(json.dumps(block['visualization']), block['location']), self._you['obj_id'])
            exists = False
            for b in self._world_state['found_blocks']:
                if b['visualization'] == block['visualization'] and b['location'] == block['location']:
                    exists = True
                    break
            if not exists:
                self._world_state['found_blocks'].append({'visualization': block['visualization'], 'location': block['location'], 'by': self._you['obj_id']})
        pass

    def filter_bw4t_observations(self, state):
        return state

    def decide_on_bw4t_action(self, state: State):
        agent_name = state[self.agent_id]['obj_id']
        self._you = state[self.agent_id]
        # Add team members
        for member in state['World']['team_members']:
            index = state['World']['team_members'].index(member)
            if self._world_state['agent_index'] == -1 and member == agent_name:
                self._world_state['agent_index'] = index
            if member != agent_name and member not in self._world_state['teammembers']:
                self._world_state['teammembers'][member] = {'state':{'type':None}, 'carrying': [], 'index':index}
                
        if len(self._world_state['goals']) == 0:
            # init goals
            self.init_goals(state)

        # Process messages from team members
        receivedMessages = self._processMessages(self._world_state['teammembers'].keys())
        self._handleMessages(state)
        # handle observations, what blocks you currently see, teammates and their actual states, doors etc
        observations = self._processObservations(state)
        self._handleObservations(observations)
        # Update trust beliefs for team members
        if not self._beliefs:
            for member in state['World']['team_members']:
                self._beliefs[member] = {
                    'trust': 0.5,
                    'competence': 0.5,
                    'willingness': 0.5
                }

        self._trustBelief(self._teamMembers, receivedMessages)

        closedDoors = [door for door in state.values()
                       if 'class_inheritance' in door and 'Door' in door['class_inheritance'] and not door['is_open']]

        # if a teammate with a higher 'dominance', aka index, is searching the same room as us move to another possible room
        for member_state in self._world_state['teammembers'].values():
            if (member_state['index'] < self._world_state['agent_index']
                    and member_state['state']['type'] is not None
                    and self._current_state == member_state['state']):
                self._phase = Phase.PLAN_NEXT_ACTION
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
                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    return action, {}
                # room is completely searched
                self._world_state['searched_rooms'].append({'room_id': self._current_room['room_name'], 'by': agent_name})
                self._phase = Phase.PLAN_NEXT_ACTION

            if Phase.PICKUP_BLOCK == self._phase:
                action = self.pickup_block(agent_name, observations)
                if action is not None:
                    return action

            if Phase.DROP_BLOCK == self._phase:
                return self.drop_block(agent_name, state)

    def plan_next_action(self, state):
        self._navigator.reset_full()
        moving_to_target = False
        missing_goals = self.get_missing_goals()
        if (len(state.get_self()['is_carrying']) < self._carrying_capacity
                and len(state.get_self()['is_carrying']) < len(missing_goals)):
            carrying_index = len(state.get_self()['is_carrying'])
            for block in self._world_state['found_blocks']:
                if block['visualization'] == missing_goals[carrying_index]['visualization']:
                    self._phase = Phase.PLAN_PATH_TO_BLOCK
                    moving_to_target = True
                    break
                    
        elif len(state[self.agent_id]['is_carrying']) > 0:
            self._phase = Phase.PLAN_PATH_TO_GOAL
            moving_to_target = True
            
        if not moving_to_target:
            self._phase = Phase.PLAN_PATH_TO_ROOM

    def plan_path_to_room(self, agent_name, state):
        # we will find and create a path to a room that is not yet searched
        # get a list of all rooms that are not yet traversed/searched (completely)
        searched_rooms = list(map(lambda e: e['room_id'], self._world_state['searched_rooms']))
        possible_rooms = [room for room in state.values()
                          if 'class_inheritance' in room and 'Door' in room['class_inheritance'] and room[
                              'room_name'] not in searched_rooms]
        
        if len(possible_rooms) == 0:
            # some rooms where not completely searched apparently, so restart searching all rooms
            possible_rooms = [room for room in state.values()
                              if 'class_inheritance' in room and 'Door' in room['class_inheritance']]
            
        # find closest room
        move_to_room = min(possible_rooms, key=lambda r: self.dist(self._you, r, state=state))
        self._sendMessage('Moving to {}'.format(move_to_room['room_name']), agent_name)
        self._current_state = {'type': State.MOVING_TO_ROOM, 'room_id': move_to_room['room_name']}
        self._current_room = move_to_room
        if not move_to_room['is_open']:
            self._door = move_to_room
            self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR
        else:
            self._navigator.add_waypoints([move_to_room['location']])
            self._phase = Phase.FOLLOW_PATH_TO_ROOM

    def plan_path_to_closed_door(self, agent_name, closedDoors, state):
        if self._door == None:
            self._door = min(closedDoors, key=lambda d: self.dist(self._you, d, state=state))
        else:
            doorLoc = self._door['location']
            # Location in front of door is south from door
            doorLoc = doorLoc[0], doorLoc[1] + 1
            # Send message of current action
            # self._sendMessage('Moving to door of ' + self._door['room_name'], agent_name)
            self._navigator.add_waypoints([doorLoc])
            self._phase = Phase.FOLLOW_PATH_TO_CLOSED_DOOR

    def plan_path_to_block(self, state):
        possible_target = self.get_closest_possible_target(state)
        if possible_target is None:
            self._phase = Phase.PLAN_PATH_TO_ROOM
        else:
            self._navigator.reset_full()
            self._navigator.add_waypoints([possible_target['location']])
            self._sendMessage(
                'Picking up goal block {} at location {}'.format(json.dumps(possible_target['visualization']), possible_target['location']),
                self._you['obj_id'])
            self._phase = Phase.FOLLOW_PATH_TO_BLOCK

    def plan_path_to_goal(self, state):
        if len(state.get_self()['is_carrying']) == 0:
            self._phase = Phase.PLAN_NEXT_ACTION
        else:
            block = state.get_self()['is_carrying'][0]
            goals = [goal for goal in self._world_state['goals']
                        if goal['visualization']['size'] == block['visualization']['size']
                            and goal['visualization']['shape'] == block['visualization']['shape']
                            and goal['visualization']['colour'] == block['visualization']['colour']]
            target = None
            for goal in goals:
                target = goal
                if not goal['satisfied']:
                    break
            targetLoc = target['location']
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
        self._navigator.add_waypoints(map(lambda e: e['location'], room_tiles))
        self._sendMessage('Searching through {}'.format(self._current_room['room_name']), agent_name)
        self._phase = Phase.SEARCH_ROOM

    def pickup_block(self, agent_name, observations):
        # TODO: update trust if block does not exist
        self._phase = Phase.PLAN_NEXT_ACTION
        missing_goals = self.get_missing_goals()
        if missing_goals is None:
            return
        for block in observations['blocks']:
            for goal in missing_goals:
                if block['visualization'] == goal['visualization']:
                    return GrabObject.__name__, {'object_id': block['obj_id']}
        blocks_that_should_be_at_location = [block for block in self._world_state['found_blocks']
                                                if block['location'] == self._you['location']]
        for block in blocks_that_should_be_at_location:
            self._world_state['found_blocks'].remove(block)

    def drop_block(self, agent_name, state):
        self._phase = Phase.PLAN_NEXT_ACTION
        block = state.get_self()['is_carrying'][0]
        block_vis = {'size':block['visualization']['size'],'shape':block['visualization']['shape'],'colour':block['visualization']['colour']}
        for goal in self._world_state['goals']:
            if block['location'] == goal['location'] and block_vis == goal['visualization']:
                goal['satisfied'] = True
                goal['verified'] = True
                goal['by'] = agent_name
                break
        self._sendMessage('Dropped goal block {} at drop location {}'.format(json.dumps(block_vis), block['location']), agent_name)
        return DropObject.__name__, {'object_id': block['obj_id']}

    def target_missing_and_at_goal(self, block, location):
        return str(block) == str(self._missing_blocks[self._current_target_block]['block']) and str(location) == str(
            self._missing_blocks[self._current_target_block]['location'])

    def init_goals(self, state):
        blocks = copy.deepcopy([{'visualization':tile['visualization'], 'location':tile['location']} for tile in state.values()
                                if 'class_inheritance' in tile
                                and 'GhostBlock' in tile['class_inheritance']
                                ])
        
        for block in blocks:
            block['visualization'].pop('opacity')
            block['visualization'].pop('visualize_from_center')
            block['visualization'].pop('depth')
        
        for i in range(len(blocks)):
            blocks[i]['index'] = i
            blocks[i]['satisfied'] = False
            blocks[i]['verified'] = False
            blocks[i]['by'] = None
            self._world_state['goals'].append(blocks[i])

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
        # TODO: take colorblind into account
        next_goal = self.get_missing_goals()[0]
        possible_blocks = [block for block in self._world_state['found_blocks']
                            if block['visualization'] == next_goal['visualization']]
        closest_target = sorted(possible_blocks, key=lambda b: self.dist(self._you, b, state))[0]
        return closest_target
    
    def get_missing_goals(self):
        missing_goals = sorted([goal for goal in self._world_state['goals']
                            if not goal['satisfied']], key=lambda g: g['index'])
        if len(missing_goals) == 0:
            # all goals are supposedly satisfied
            # TODO: handle case where supposedly all goals are SAT
            return None
        else:
            return missing_goals
