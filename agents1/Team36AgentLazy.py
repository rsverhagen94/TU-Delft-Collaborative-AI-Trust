from .Team36BaseAgent import BaseAgent
from typing import Dict


class LazyAgent(BaseAgent):

    def __init__(self, settings: Dict[str, object]):
        super().__init__(settings)

    def initialize(self):
        super().initialize()
        self._is_lazy = True