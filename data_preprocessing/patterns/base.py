from abc import ABC, abstractmethod


class BasePattern(ABC):
    semantic_type: str

    @abstractmethod
    def detect(self, values) -> float:
        """
        Input: iterable of sample column values (strings or numbers)
        Output: confidence score between 0 and 1
        """
        pass

    @abstractmethod
    def normalize(self, values):
        """
        Input: full pandas Series
        Output: pandas Series with ALL values converted
                into ONE canonical format
        """
        pass
