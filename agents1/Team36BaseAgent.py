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
    SEARCH_ROOM = 13,
    # TODO: remove when deemed absolutely unnecessary/useless
    # FOLLOW_PATH_TO_VERIFY_GOAL = 14,
    # VERIFY_GOAL = 15,
    # PLAN_MOVE_OFF_GOAL = 16,
    # MOVE_OFF_GOAL = 17
    PLAN_VERIFY_GOALS = 18,
    VERIFY_GOALS = 19,
    PLAN_FIX_SOLUTION = 20,
    FIX_SOLUTION = 21




class State(enum.Enum):
    MOVING_TO_ROOM = 0,
    MOVING_TO_GOAL = 1,
    PICKING_UP_BLOCK = 2,

class Belief(enum.Enum):
    TRUST = 0,
    COMPETENCE = 1,
    WILLINGNESS = 2

# TODO: fix missing information for block visualization (like for colour blind) or additional (useless) information, cuz who knows what whacky stuff other agents do
class BaseAgent(BaseLineAgent):

    def __init__(self, settings: Dict[str, object]):
        super().__init__(settings)

        self._maxTrust = 1.0
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

        self.__next_phase = []

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
        self._beliefs = {}
        self._test = False

        self._world_state = {
            'found_blocks': [],     # list of blocks, contains {'location','visualization','by',('obj_id')}
            'teammembers': {},      # dict of teammembers accessible by their id, contains {'state','carrying','index'}
            'goals': [],            # list of goals, contains {'location','visualization','index','satisfied','by','checked'}
            'searched_rooms': [],   # list of rooms, contains {'room_id','by'}
            'opened_doors': [],     # list of doors, contains {'room_id','by'}
            'agent_index': -1,      # index/dominance of this agent
        }

        self.__next_phase = []

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
                    current_world_state['opened_doors'].append({'room_id': room_id, 'by': member, 'checked': False})

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
                    # add info to teammember
                    # current_world_state['teammembers'][member]['state'] = {'type': State.PICKING_UP_BLOCK, 'block': {'visualization': block_vis, 'location': block_loc}}
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
                        self._decreaseBelief(Belief.TRUST, member, 0.2)
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
                    block_vis = json.loads(vis_and_loc[0])
                    block_loc = eval(vis_and_loc[1])
                    for goal in self._world_state['goals']:
                        if goal['visualization'] == block_vis and not block_loc == goal['location'] and not goal['satisfied'] and not self.previousGoalsSatisfied(goal['index'] - 1):
                            self._decreaseBelief(Belief.WILLINGNESS, member, 0.1)

                if 'Found' in message and 'colour' not in message:
                    #update capability (agent does not provide all information)
                    self._decreaseBelief(Belief.COMPETENCE, member, 0.1)

    def previousGoalsSatisfied(self, goalIndex):
        while goalIndex >= 0:
            for goal in self._world_state['goals']:
                if goal['index'] == goalIndex and goal['satisfied']:
                    goalIndex -= 1
                    break
                else:
                    return False
        return True

    def _increaseBelief(self, type, member, amount):
        if type == Belief.WILLINGNESS:
            if self._beliefs[member][Belief.WILLINGNESS] + amount < self._maxTrust:
                self._beliefs[member][Belief.WILLINGNESS] += amount
            else:
                self._beliefs[member][Belief.WILLINGNESS] = self._maxTrust
        elif type == Belief.TRUST:
            if self._beliefs[member][Belief.TRUST] + amount < self._maxTrust:
                self._beliefs[member][Belief.TRUST] += amount
            else:
                self._beliefs[member][Belief.TRUST] = self._maxTrust
        elif type == Belief.COMPETENCE:
            if self._beliefs[member][Belief.COMPETENCE] + amount < self._maxTrust:
                self._beliefs[member][Belief.COMPETENCE] += amount
            else:
                self._beliefs[member][Belief.COMPETENCE] = self._maxTrust

    def _decreaseBelief(self, type, member, amount):
        if type == Belief.WILLINGNESS:
            if self._beliefs[member][Belief.WILLINGNESS] - amount > 0.0:
                self._beliefs[member][Belief.WILLINGNESS] -= amount
            else:
                self._beliefs[member][Belief.WILLINGNESS] = 0.0
        elif type == Belief.TRUST:
            if self._beliefs[member][Belief.TRUST] - amount > 0.0:
                self._beliefs[member][Belief.TRUST] -= amount
            else:
                self._beliefs[member][Belief.TRUST] = 0.0
        elif type == Belief.COMPETENCE:
            if self._beliefs[member][Belief.COMPETENCE] - amount > 0.0:
                self._beliefs[member][Belief.COMPETENCE] -= amount
            else:
                self._beliefs[member][Belief.COMPETENCE] = 0.0

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
        doors = [door for door in state.values()
                       if 'class_inheritance' in door and 'Door' in door['class_inheritance']]
        for door in doors:
            observations['doors'].append({'room_name': door['room_name'], 'is_open': door['is_open'], 'obj_id': door['obj_id']})

        return observations
    
    def _handleObservations(self, observations):
        blocks_that_should_be_at_location = [block for block in self._world_state['found_blocks']
                                                if self.dist(self._you, block) <= 0]
        for block in observations['blocks']:
            # if block is on a correct goal position, skip
            goals = [g for g in self._world_state['goals']
                        if g['visualization'] == block['visualization'] and g['location'] == block['location']]
            if len(goals) > 0:
                # this block is placed on a correct goal position
                continue

            self._sendMessage('Found goal block {} at location {}'.format(json.dumps(block['visualization']), block['location']), self._you['obj_id'])
            exists = False
            for b in self._world_state['found_blocks']:
                if b['visualization'] == block['visualization'] and b['location'] == block['location']:
                    exists = True
                    break
            if not exists:
                self._world_state['found_blocks'].append({'visualization': block['visualization'], 'location': block['location'], 'by': self._you['obj_id']})

        # remove all blocks that should exist, but don't and update trust
        for block in blocks_that_should_be_at_location:
            verified = False
            for b in observations['blocks']:
                if b['visualization'] == block['visualization'] and b['location'] == block['location']:
                    verified = True
                    self._increaseBelief(Belief.TRUST, block['by'], 0.05)
                    break
            if not verified:
                self._world_state['found_blocks'].remove(block)
                self._decreaseBelief(Belief.TRUST, block['by'], 0.1)

        # handle goal verification (only done if standing on a goal, since sense capabilities is a bit iffy)
        goals_to_verify_at_cur_location = [goal for goal in self._world_state['goals']
                                    if goal['satisfied']
                                        and not goal['verified']
                                        and goal['location'] == self._you['location']]
        if len(goals_to_verify_at_cur_location) > 0:
            goal = goals_to_verify_at_cur_location[0]
            correct_blocks_on_goal = [b for b in observations['blocks']
                                        if b['location'] == goal['location']
                                            and b['visualization'] == goal['visualization']]
            if len(correct_blocks_on_goal) > 0:
                # this goal has a correct block
                goal['verified'] = True
                self._increaseBelief(Belief.TRUST, goal['by'], 0.2)
            else:
                # this goal was not satisfied, update trust of goal['by']
                goal['satisfied'] = False
                self._decreaseBelief(Belief.TRUST, goal['by'], 0.3)

        # handle visible doors and update trust
        for door in observations['doors']:
            for d in self._world_state['opened_doors']:
                if not door['is_open'] and door['room_name'] == d['room_id'] and not d['checked']:
                    # door that was said to be opened is not open, decrease trust
                    self._decreaseBelief(Belief.TRUST, d['by'], 0.05)
                    d['checked'] = True
                if door['is_open'] and door['room_name'] == d['room_id'] and not d['checked']:
                    # member opened door when saying it would open door, increase trust
                    self._increaseBelief(Belief.TRUST, d['by'], 0.01)
                    d['checked'] = True

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
                    Belief.TRUST: 0.5,
                    Belief.COMPETENCE: 0.5,
                    Belief.WILLINGNESS: 0.5
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
                if len(self.__next_phase) > 0:
                    self._phase = self.__next_phase.pop()
                else:
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
                    if len([goal for goal in self._world_state['goals'] if goal['satisfied'] and not goal['verified']]) > 0:
                        # if there exist unverified but satisfied goals, verify those
                        self.__next_phase.append(Phase.PLAN_VERIFY_GOALS)

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

                next_goal_index = len(state.get_self()['is_carrying'])
                missing_goals = self.get_missing_goals()
                if len(missing_goals) > next_goal_index:
                    goal = missing_goals[next_goal_index]
                    needed_goal_blocks = [b for b in observations['blocks']
                                            if b['visualization'] == goal['visualization']]
                    if len(needed_goal_blocks) > 0:
                        # we have a block we can pick up
                        self._phase = Phase.PICKUP_BLOCK
                        self.__next_phase.append(Phase.PLAN_SEARCH_ROOM)
                        continue

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
                action = self.drop_block(agent_name, state)
                if action is not None:
                    return action

            # TODO: remove when deemed absolutely unnecessary/useless
            # if Phase.FOLLOW_PATH_TO_VERIFY_GOAL == self._phase:
            #     self._state_tracker.update(state)
            #     action = self._navigator.get_move_action(self._state_tracker)
            #     if action != None:
            #         return action, {}
            #     self._phase = Phase.VERIFY_GOAL
            #
            # if Phase.VERIFY_GOAL == self._phase:
            #     self.verify_goal(observations)
            #
            # if Phase.PLAN_MOVE_OFF_GOAL == self._phase:
            #     self.plan_move_off_goal()
            #
            # if Phase.MOVE_OFF_GOAL == self._phase:
            #     self._state_tracker.update(state)
            #     action = self._navigator.get_move_action(self._state_tracker)
            #     if action != None:
            #         return action, {}
            #     self._phase = Phase.DROP_BLOCK

            if Phase.PLAN_VERIFY_GOALS == self._phase:
                goals_to_verify = [goal for goal in self._world_state['goals'] if goal['satisfied'] and not goal['verified']]
                self._navigator.reset_full()
                for goal in goals_to_verify:
                    self._navigator.add_waypoint(goal['location'])
                self._phase = Phase.VERIFY_GOALS

            if Phase.VERIFY_GOALS == self._phase:
                # handle observations has a check to see if you are on a goal state that needs verification
                # if so it verifies it or not
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action is not None:
                    return action, {}
                if len(self.get_missing_goals()) == 0:
                    self._phase = Phase.PLAN_FIX_SOLUTION
                else:
                    self._phase = Phase.PLAN_NEXT_ACTION

            if Phase.PLAN_FIX_SOLUTION == self._phase:
                # TODO:
                pass

            if Phase.FIX_SOLUTION == self._phase:
                # TODO:
                pass

    def plan_next_action(self, state):
        self._navigator.reset_full()
        moving_to_target = False
        missing_goals = self.get_missing_goals()
        if len(missing_goals) == 0:
            self._phase = Phase.PLAN_VERIFY_GOALS
            return
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
            self._current_state = {'type': State.PICKING_UP_BLOCK, 'block': {'visualization': possible_target['visualization'], 'location': possible_target['location']}}
            self._phase = Phase.FOLLOW_PATH_TO_BLOCK

    def plan_path_to_goal(self, state):
        if len(state.get_self()['is_carrying']) == 0:
            self._phase = Phase.PLAN_NEXT_ACTION
        else:
            block_vis = copy.copy(state.get_self()['is_carrying'][0]['visualization'])
            block_vis.pop('opacity')
            block_vis.pop('visualize_from_center')
            block_vis.pop('depth')
            goal = None
            for g in self.get_missing_goals():
                if g['visualization'] == block_vis:
                    goal = g
                    break
            if g is None:
                self._phase = Phase.DROP_BLOCK
            else:
                self._navigator.reset_full()
                self._navigator.add_waypoint(goal['location'])
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
        self._phase = Phase.PLAN_NEXT_ACTION
        missing_goals = self.get_missing_goals()
        next_goal_index = len(self._you['is_carrying'])
        if len(missing_goals) <= next_goal_index:
            return
        goal = missing_goals[next_goal_index]
        for block in observations['blocks']:
            if block['visualization'] == goal['visualization']:
                # remove block from world_state
                for b in self._world_state['found_blocks']:
                    if b['visualization'] == block['visualization'] and b['location'] == block['location']:
                        self._world_state['found_blocks'].remove(b)
                        break

                self._sendMessage(
                    'Picking up goal block {} at location {}'.format(json.dumps(block['visualization']),
                                                                     block['location']),
                    agent_name)
                return GrabObject.__name__, {'object_id': block['obj_id']}

    def drop_block(self, agent_name, state):
        self._phase = Phase.PLAN_NEXT_ACTION
        if len(state.get_self()['is_carrying']) == 0:
            return
        block = state.get_self()['is_carrying'][0]
        block_vis = {'size':block['visualization']['size'],'shape':block['visualization']['shape'],'colour':block['visualization']['colour']}
        for goal in self.get_missing_goals():
            if state.get_self()['location'] == goal['location'] and block_vis == goal['visualization']:
                goal['satisfied'] = True
                goal['verified'] = True
                goal['by'] = agent_name
                break
        if len(state.get_self()['is_carrying']) > 1:
            self.__next_phase.append(Phase.PLAN_PATH_TO_GOAL)
        self._sendMessage('Dropped goal block {} at drop location {}'.format(json.dumps(block_vis), state.get_self()['location']), agent_name)
        return DropObject.__name__, {'object_id': block['obj_id']}

    # TODO: remove when deemed absolutely unnecessary/useless
    # def verify_goal(self, observations):
    #     found_goal = False
    #     for goal in self._world_state['goals']:
    #         for block in observations['blocks']:
    #             if block['visualization'] == goal['visualization'] and block['location'] == goal['location']:
    #                 goal['verified'] = True
    #                 found_goal = True
    #                 self._increaseBelief(Belief.TRUST, goal['by'], 0.2)
    #                 self._phase = Phase.PLAN_PATH_TO_GOAL
    #     if not found_goal:
    #         for goal in self._world_state['goals']:
    #             if not goal['verified'] and goal['satisfied']:
    #                 goal['satisfied'] = False
    #                 self._decreaseBelief(Belief.TRUST, goal['by'], 0.3)
    #         self._phase = Phase.PLAN_MOVE_OFF_GOAL
    #
    # def plan_move_off_goal(self):
    #     current_location = self._you['location']
    #     next_location = current_location[0] - 1, current_location[1] - 1
    #     self._navigator.reset_full()
    #     self._navigator.add_waypoint(next_location)
    #     self._phase = Phase.MOVE_OFF_GOAL

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
        if state is None:
            return abs(start['location'][0] - target['location'][0]) + abs(start['location'][1] - target['location'][1])
        else:
            # calculate grassfire heuristic, will only calculate the field if necessary, aka pieces/you moved
            self.create_gf_field(state)
            return self.gf_field[target['location'][0]][target['location'][1]]

    def get_closest_possible_target(self, state):
        # TODO: take colorblind into account
        next_goal_index = len(state.get_self()['is_carrying'])
        next_goal = self.get_missing_goals()[next_goal_index]
        possible_blocks = [block for block in self._world_state['found_blocks']
                            if block['visualization'] == next_goal['visualization']]
        closest_target = sorted(possible_blocks, key=lambda b: self.dist(self._you, b, state))[0]
        return closest_target
    
    def get_missing_goals(self):
        missing_goals = sorted([goal for goal in self._world_state['goals']
                            if not goal['satisfied']], key=lambda g: g['index'])
        return missing_goals
