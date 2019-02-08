from .network import BaseLayer


class Lambda(BaseLayer):
    def __init__(self, fn):
        super(Lambda, self).__init__()
        self._fn = fn

    def call(self, inputs):
        return self._fn(inputs)
