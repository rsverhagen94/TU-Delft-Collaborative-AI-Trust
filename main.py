#from agents1.StrongAgent import StrongAgent
from agents1.FinalAgent import ColorblindAgent, LiarAgent, LazyAgent, StrongAgent
#from agents1.LiarAgent import LiarAgent
#from agents1.LazyAgent import LazyAgent
from bw4t.BW4TWorld import BW4TWorld
from bw4t.statistics import Statistics
#from agents1.BW4TBaselineAgent import BaseLineAgent
#from agents1.BW4THuman import Human


"""
This runs a single session. You have to log in on localhost:3000 and
press the start button in god mode to start the session.
"""

if __name__ == "__main__":
    agents = [
        {'name':'Liar', 'botclass':LiarAgent, 'settings':{}},
        #{'name': 'agent2', 'botclass': StrongAgent, 'settings': {}},
        # {'name': 'Strong', 'botclass': StrongAgent, 'settings': {}},
        {'name':'Lazy', 'botclass':LazyAgent, 'settings':{}},
        {'name': 'Strong', 'botclass': StrongAgent, 'settings': {}},
        # {'name': 'Strong1', 'botclass': StrongAgent, 'settings': {}},
        # {'name':'agent4', 'botclass':LazyAgent, 'settings':{}},
        {'name':'ColorblindAgent', 'botclass':ColorblindAgent, 'settings':{}},
        # {'name':'agent2', 'botclass':BaseLineAgent, 'settings':{}},
        # {'name':'human', 'botclass':Human, 'settings':{}}
        ]

    print("Started world...")
    world=BW4TWorld(agents).run()
    print("DONE!")
    print(Statistics(world.getLogger().getFileName()))
