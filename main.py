from bw4t.BW4TWorld import BW4TWorld
from bw4t.statistics import Statistics
from agents1.BW4TBaselineAgent import BaseLineAgent
from agents1.Team09Agent import StrongAgent
from agents1.BW4THuman import Human


"""
This runs a single session. You have to log in on localhost:3000 and 
press the start button in god mode to start the session.
"""

if __name__ == "__main__":
    agents = [
        {'name':'agent1', 'botclass': StrongAgent, 'settings': {'slowdown':10}},
        {'name':'agent2', 'botclass': StrongAgent, 'settings': {}},
        {'name': 'agent3', 'botclass': StrongAgent, 'settings': {'slowdown': 10}},
        {'name': 'agent4', 'botclass': StrongAgent, 'settings': {}},
        {'name': 'agent5', 'botclass': StrongAgent, 'settings': {'slowdown': 10}},
        {'name': 'agent6', 'botclass': StrongAgent, 'settings': {}},
        {'name': 'agent7', 'botclass': StrongAgent, 'settings': {'slowdown': 10}},
        {'name': 'agent8', 'botclass': StrongAgent, 'settings': {}},
        {'name': 'agent9', 'botclass': StrongAgent, 'settings': {'slowdown': 10}},
        {'name': 'agent10', 'botclass': StrongAgent, 'settings': {}},
        {'name': 'agent11', 'botclass': StrongAgent, 'settings': {'slowdown': 10}},
        {'name': 'agent12', 'botclass': StrongAgent, 'settings': {}},
        {'name': 'agent13', 'botclass': StrongAgent, 'settings': {'slowdown': 10}},
        {'name': 'agent14', 'botclass': StrongAgent, 'settings': {}},
        ]

    print("Started world...")
    world=BW4TWorld(agents).run()
    print("DONE!")
    print(Statistics(world.getLogger().getFileName()))
