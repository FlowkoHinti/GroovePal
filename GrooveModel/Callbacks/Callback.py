import logging

from GrooveModel.Learner.LearnerState import LearnerState


class Callback:
    """Base class for all callbacks."""

    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger("CallbackLogger")
        self.state = {}  # shared between cbs will be replaced with shared state via attach_state()

    def attach_state(self, shared_state: dict):
        """Attach the shared state dictionary from CallbackManager."""
        self.state = shared_state

    def on_train_begin(self, learner: LearnerState): pass

    def on_train_end(self, learner: LearnerState): pass

    def on_epoch_begin(self, learner: LearnerState): pass

    def on_epoch_end(self, learner: LearnerState): pass

    def on_batch_begin(self, learner: LearnerState): pass

    def on_batch_end(self, learner: LearnerState): pass

    def on_after_backward(self, learner: LearnerState): pass


class CallbackManager:
    """Manages a list of callbacks and shared state between them."""

    def __init__(self, callbacks):
        self.callbacks = callbacks
        self.state = {}
        # Attach shared state once
        for cb in self.callbacks:
            if hasattr(cb, "attach_state"):
                cb.attach_state(self.state)
            else:
                cb.state = self.state  # fallback

    def call(self, method, *args, **kwargs):
        for cb in self.callbacks:
            getattr(cb, method)(*args, **kwargs)
