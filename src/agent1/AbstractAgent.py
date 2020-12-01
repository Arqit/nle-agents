class AbstractAgent:
    """
    AbstractAgent

    """

    def __init__(self, **kwargs):
        raise NotImplementedError()

    def act(self, observation):
        raise NotImplementedError()
