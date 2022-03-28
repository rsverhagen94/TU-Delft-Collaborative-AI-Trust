from agents1.Team36AgentLiarr import Liar
from agents1.Team36AgentLazy import Lazy
from agents1.Team36BaseAgent import BaseAgent
from bw4t.BW4TWorld import BW4TWorld
from bw4t.statistics import Statistics
from agents1.BW4TBaselineAgent import BaseLineAgent
from agents1.BW4THuman import Human
from agents1.Team36ColorblindAgent import ColorBlindAgent


"""
This runs a single session. You have to log in on localhost:3000 and 
press the start button in god mode to start the session.
"""

if __name__ == "__main__":
    agents = [

        # {'name':'agent1', 'botclass':BaseLineAgent, 'settings':{'slowdown':10}},
        # {'name':'agent2', 'botclass':BaseLineAgent, 'settings':{}},
        {'name':'liar', 'botclass':BaseAgent, 'settings':{}},
        {'name':'liar2', 'botclass':BaseAgent, 'settings':{}},
        {'name':'lazy3', 'botclass':BaseAgent, 'settings':{}},
        {'name':'lazy4', 'botclass':BaseAgent, 'settings':{}},
        # {'name':'human', 'botclass':Human, 'settings':{}}
        ]

    print("Started world...")
    world=BW4TWorld(agents).run()
    print("DONE!")
    print(Statistics(world.getLogger().getFileName()))
