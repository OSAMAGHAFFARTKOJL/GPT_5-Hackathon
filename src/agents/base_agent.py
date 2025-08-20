from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging

class BaseAgent(ABC):
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        pass
    
    def validate_input(self, state: Dict[str, Any]) -> bool:
        return True
    
    def log_execution(self, action: str, details: str = ""):
        self.logger.info(f"{self.name} - {action}: {details}")