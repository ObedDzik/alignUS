class EventEmitter: 
    """Simple interface for registering and emitting events.
    
    Programs can use this class to define custom events and register callbacks to be executed when the event is emitted.
    """
    def __init__(self, *args, **kwargs): 
        super().__init__(*args, **kwargs)
        self._callbacks = {}

    def add_callback(self, event, callback): 
        self._callbacks.setdefault(event, []).append(callback)

    def emit_event(self, event, *args, **kwargs): 
        for callback in self._callbacks.get(event, []): 
            callback(*args, **kwargs)

    def on_event(self, event): 
        def wrapper(fn): 
            self.add_callback(event, fn)
            return fn
        return wrapper
