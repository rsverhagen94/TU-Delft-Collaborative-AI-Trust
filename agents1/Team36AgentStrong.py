import copy
from operator import le
from typing import final, List, Dict, Final
from functools import reduce
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
    PLAN_NEXT_ACTION=0,
    PLAN_PATH_TO_CLOSED_DOOR=1,
    PLAN_PATH_TO_ROOM=2,
    PLAN_PATH_TO_BLOCK=3,
    PLAN_PATH_TO_GOAL=4,
    PLAN_SEARCH_ROOM=5
    FOLLOW_PATH_TO_CLOSED_DOOR=6,
    FOLLOW_PATH_TO_ROOM=7,
    FOLLOW_PATH_TO_BLOCK=8,
    FOLLOW_PATH_TO_GOAL=9,
    OPEN_DOOR=10,
    PICKUP_BLOCK=11,
    DROP_BLOCK=12,
    SEARCH_ROOM=13
    
class State(enum.Enum):
    MOVING_TO_ROOM=0,
    MOVING_TO_GOAL=1,
   


# TODO: message sending
# TODO: message handling
# TODO: keep track of status (carrying, color, room, etc) of team members
class StrongAgent(BaseLineAgent):

    def __init__(self, settings:Dict[str,object]):
        super().__init__(settings)

    def initialize(self):
        super().initialize()
        self._carrying = []
        self._unsearched_rooms = None
        self._rooms_searched_by_teammembers = []
        self._required_blocks = None
        self._found_blocks = []
        self._current_room = None  
        self._door = None 
        self._phase = Phase.PLAN_NEXT_ACTION 
        self.gf_start = None
        self._teammember_states = {}    # states of teammembers will be added in this dict with the key being the teammembers ID
        self._current_state = {'type': None}
        self._index = -1
        
    def _handleMessages(self, messages):
        for member in self._teamMembers:
            if member not in messages:
                continue
            for msg in messages[member]:
                if 'Moving to ' in msg and 'door' not in msg:
                    room_id = msg.replace("Moving to ", "", 1)
                    self._rooms_searched_by_teammembers.append(room_id)
                    self._teammember_states[member]['state'] = {'type': State.MOVING_TO_ROOM, 'room': room_id}

    def filter_bw4t_observations(self, state):
        return state

    def decide_on_bw4t_action(self, state:State):
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
            if member!=agent_name and member not in self._teamMembers:
                self._teamMembers.append(member)   
        # add required blocks
        if self._required_blocks == None:
            self._required_blocks = copy.deepcopy(sorted([target for target in state.values()
                                        if 'drop_zone_nr' in target and target['is_goal_block'] == True], key=lambda target: target['drop_zone_nr']))
            print("required blocks: ")
            for required in self._required_blocks:
                required['visualization'].pop('depth')
                required['visualization'].pop('opacity')
                required['visualization'].pop('visualize_from_center')
                print(required['visualization'])
            
        # initialize unsearched rooms list
        if self._unsearched_rooms == None:
            self._unsearched_rooms = state.get_all_room_names()
        # Process messages from team members
        receivedMessages = self._processMessages(self._teamMembers)
        self._handleMessages(receivedMessages)
        # Update trust beliefs for team members
        self._trustBlief(self._teamMembers, receivedMessages)
        
        # handle state, process what you see in the current state, thus blocks and player actions
        # TODO: handle teammate states (namely for trust)
        # if we find a new target append it to the currently found blocks
        _targets = copy.deepcopy([target for target in state.values() 
                        if 'is_collectable' in target and target['is_collectable']
                            and len(target['carried_by']) == 0])
        for target in _targets:
            target['visualization'].pop('opacity')
            target['visualization'].pop('depth')
            target['visualization'].pop('visualize_from_center')
        found_target_ids = map(lambda a: a['obj_id'], self._found_blocks)
        for target in _targets:
            for b in self._required_blocks:
                if b['visualization'] == target['visualization']:
                    self._sendMessage('Found goal block {} at location {}'.format(target['visualization'], target['location']), agent_name)
            is_found = target['obj_id'] in found_target_ids
            if not is_found:
                self._found_blocks.append(target)
            else:
                # find already existant target block and if the location has changed update the block
                for i in range(len(self._found_blocks)):
                    if (self._found_blocks[i]['obj_id'] == target['obj_id']
                        and self._found_blocks[i]['location'] != target['location']):
                        self._found_blocks.pop(i)
                        self._found_blocks.append(target)
                        break
        
        closedDoors = [door for door in state.values()
            if 'class_inheritance' in door and 'Door' in door['class_inheritance'] and not door['is_open']]
        
        # if a teammate with a higher 'dominance', aka index, is searching the same room as us move to another possible room
        for member_states in self._teammember_states.values():
            # print("member index: {}, my index: {}".format(member_states['index'], self._index))
            # print(str(member_states['state']))
            # print(str(self._current_state))
            if (member_states['index'] < self._index 
                    and member_states['state']['type'] == State.MOVING_TO_ROOM 
                    and self._current_state == member_states['state']):
                self._phase = Phase.PLAN_NEXT_ACTION
                print("NEED TO PICK ANOTHER ROOM")
                break
        
        while True:
            if Phase.PLAN_NEXT_ACTION==self._phase:
                self._navigator.reset_full()
                moving_to_target = False
                if len(self._carrying) < 2 and len(self._carrying) < len(self._required_blocks):
                    for block in self._found_blocks:
                        if self._required_blocks[0]['visualization'] == block['visualization']:
                            self._phase = Phase.PLAN_PATH_TO_BLOCK
                            moving_to_target = True
                else:
                    self._phase = Phase.PLAN_PATH_TO_GOAL
                    moving_to_target = True
                            
                if not moving_to_target:
                    self._phase = Phase.PLAN_PATH_TO_ROOM
            
            if Phase.PLAN_PATH_TO_ROOM==self._phase:
                # we will find and create a path to a room that is not yet searched
                # get a list of all rooms that are not yet traversed/searched (completely)
                possible_rooms = [room for room in state.values()
                                    if 'class_inheritance' in room and 'Door' in room['class_inheritance'] and room['room_name'] in self._unsearched_rooms]
                    
                # remove rooms that are already being searched by teammembers from possible roomslass_inheritance'] and room['name'] == member['state']['room']])
                possible_rooms = [room for room in possible_rooms if room['room_name'] not in self._rooms_searched_by_teammembers]
                        
                if len(possible_rooms) == 0:
                    # some rooms where not completely searched apparently, so restart searching all rooms
                    possible_rooms = [room for room in state.values()
                                    if 'class_inheritance' in room and 'Door' in room['class_inheritance']]
                    self._unsearched_rooms = state.get_all_room_names()
                    
                # find closest room
                move_to_room = min(possible_rooms, key=lambda r: self.dist(self._you, r, state=state))
                
                self._sendMessage('Moving to {}'.format(move_to_room['room_name']), agent_name)
                self._current_state = {'type': State.MOVING_TO_ROOM, 'room': move_to_room['room_name']}
                        
                self._current_room = move_to_room
                if not move_to_room['is_open']:
                    self._door = move_to_room
                    self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR
                else:
                    # TODO: send message
                    self._navigator.add_waypoints([move_to_room['location']])
                    self._phase=Phase.FOLLOW_PATH_TO_ROOM
                
            if Phase.PLAN_PATH_TO_CLOSED_DOOR==self._phase:
                if self._door == None:
                    # Pick door based on the closest manhattan distance
                    self._door = min(closedDoors, key=lambda d: self.dist(self._you, d, state=state))
                            
                doorLoc = self._door['location']
                # Location in front of door is south from door
                doorLoc = doorLoc[0],doorLoc[1]+1
                # Send message of current action
                self._sendMessage('Moving to door of ' + self._door['room_name'], agent_name)
                self._navigator.add_waypoints([doorLoc])
                self._phase=Phase.FOLLOW_PATH_TO_CLOSED_DOOR
            
            if Phase.PLAN_PATH_TO_BLOCK==self._phase:                
                possible_target = min([target for target in self._found_blocks 
                                        if self._required_blocks[0]['visualization'] == target['visualization']], key=lambda t: self.dist(self._you, t, state=state))
                if possible_target == None:
                    self._phase = Phase.PLAN_PATH_TO_ROOM
                else:
                    self._navigator.reset_full()
                    self._navigator.add_waypoints([possible_target['location']])
                    self._block = possible_target
                    self._phase = Phase.FOLLOW_PATH_TO_BLOCK
            
            if Phase.PLAN_PATH_TO_GOAL==self._phase:                
                self._navigator.reset_full()
                # get the goal for the currently carrying block
                goals = [goal for goal in state.values() 
                            if 'is_goal_block' in goal and goal['is_goal_block']
                                and goal['visualization']['shape'] == self._carrying[0]['visualization']['shape']
                                and goal['visualization']['size'] == self._carrying[0]['visualization']['size']
                                and goal['visualization']['colour'] == self._carrying[0]['visualization']['colour']]
                self._navigator.add_waypoints([goals[0]['location']])
                self._phase = Phase.FOLLOW_PATH_TO_GOAL
                
                self._current_state = {'type': State.MOVING_TO_GOAL}
            
            if Phase.FOLLOW_PATH_TO_ROOM==self._phase:
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    return action, {}
                self._phase = Phase.PLAN_SEARCH_ROOM
            
            if Phase.FOLLOW_PATH_TO_BLOCK==self._phase:
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    return action, {}
                else:
                    self._phase = Phase.PICKUP_BLOCK
            
            if Phase.FOLLOW_PATH_TO_GOAL==self._phase:
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    return action, {}
                else:
                    self._phase = Phase.DROP_BLOCK

            if Phase.FOLLOW_PATH_TO_CLOSED_DOOR==self._phase:
                self._state_tracker.update(state)
                # Follow path to door
                action = self._navigator.get_move_action(self._state_tracker)
                if action!=None:
                    return action, {}   
                self._phase=Phase.OPEN_DOOR

            if Phase.OPEN_DOOR==self._phase:
                self._phase=Phase.PLAN_SEARCH_ROOM
                # Open door
                self._sendMessage('Opening door of {}'.format(self._current_room['room_name']), agent_name)
                action = OpenDoorAction.__name__, {'object_id':self._door['obj_id']}
                self._door = None
                return action
            
            if Phase.PLAN_SEARCH_ROOM==self._phase:
                self._navigator.reset_full()
                room_tiles = [tile for tile in state.values() 
                                if 'room_name' in tile and self._current_room['room_name'] is tile['room_name']
                                    and 'AreaTile' in tile['class_inheritance'] and tile['is_traversable']]
                for tile in room_tiles:
                    self._navigator.add_waypoints([tile['location']])
                self._sendMessage('Searching through {}'.format(self._current_room['room_name']), agent_name)
                self._phase = Phase.SEARCH_ROOM
            
            if Phase.SEARCH_ROOM==self._phase:
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    return action, {}
                # room is completely searched
                self._unsearched_rooms.remove(self._current_room['room_name'])
                self._phase = Phase.PLAN_NEXT_ACTION
            
            if Phase.PICKUP_BLOCK==self._phase:
                print("picking up block: {}".format(self._block['visualization']))
                self._carrying.append(self._block)
                action = GrabObject.__name__, {'object_id': self._block['obj_id']}
                self._sendMessage('Picking up goal block {} at location {}'.format(self._block['visualization'], self._block['location']), agent_name)
                self._phase = Phase.PLAN_NEXT_ACTION
                self._block = None
                self._required_blocks.append(self._required_blocks.pop(0))
                return action
            
            if Phase.DROP_BLOCK==self._phase:
                block = self._carrying.pop(0)
                self._sendMessage('Dropped goal block {} at location {}'.format(block['visualization'], state.get_self()['location']), agent_name)
                action = DropObject.__name__, {'object_id': block['obj_id']}
                
                # we dropped a block at a goal state so remove it from requirements
                for i in reversed(range(len(self._required_blocks))):
                    if (self._required_blocks[i]['visualization']['shape'] == block['visualization']['shape'] and 
                        self._required_blocks[i]['visualization']['shape'] == block['visualization']['shape'] and 
                        self._required_blocks[i]['visualization']['shape'] == block['visualization']['shape']):
                        self._required_blocks.pop(i)
                        print("still need {} target blocks".format(len(self._required_blocks)))
                        break
                        
                
                if len(self._carrying) == 0:
                    self._phase = Phase.PLAN_NEXT_ACTION
                else:
                    self._phase = Phase.PLAN_PATH_TO_GOAL
                return action

    def create_gf_field(self, state):
        # TODO take moved pieces into account
        if self.gf_start != None and state.get_self()['location'][0] == self.gf_start[0] and state.get_self()['location'][1] == self.gf_start[1]:
            return
        
        print("creating grassfire algo field")
        map=state.get_traverse_map()
        doors = [door['location'] for door in state.values() if 'class_inheritance' in door and 'Door' in door['class_inheritance']]
        start = state.get_self()['location']
        width, length = state.get_world_info()['grid_shape']
        self.gf_field = []
        for x in range(width):
            self.gf_field.append([])
            for y in range(length):
                val = -1
                if  not map[(x,y)]:
                    val = -2
                self.gf_field[x].append(val) # this grassfire map will have -1 if not yet traversed, and -2 if it can't be traversed
                
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
            for dx in [-1,0,1]:
                x = l[0] + dx
                for dy in [-1,0,1]:
                    y = l[1] + dy
                    if dx == 0 and dy == 0:     # dont do same spot
                        pass
                    elif dx != 0 and dy != 0:   # dont do diagonals
                        pass
                    elif x < 0 or y < 0 or x >= width or y >= length:   # dont do outside of map
                        pass
                    elif self.gf_field[x][y] == -1:# or gf_field[x][y] > level+1:  # do unvisited or 
                        t = (x,y)
                        new_q.append(t)
            
            # if the current frontier is empty replace it with the new frontier and increase the distance (level)
            if len(q) == 0:
                q = new_q
                new_q = []
                level += 1
            
        self.gf_start = (state.get_self()['location'][0], state.get_self()['location'][1])
        
        
    def dist(self, start, target, state=None):
        if state == None:
            return sqrt((start['location'][0] - target['location'][0])**2 + (start['location'][1] - target['location'][1])**2)
        else:
            # calculate grassfire heuristic, will only calculate the field if necessary, aka pieces/you moved
            self.create_gf_field(state)
                
            return self.gf_field[target['location'][0]][target['location'][1]]
    