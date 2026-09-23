from collections import UserDict


class Registry(UserDict): 
    def register(self, obj):
        if callable(obj): 
            self[obj.__name__] = obj
            return obj

        def wrapper(inner_obj): 
            key = obj or inner_obj.__name__
            self[key] = inner_obj
            return inner_obj
        
        return wrapper 

    def __call__(self, name, *args, **kwargs):
        if name not in self:
            raise KeyError(f"{name} not found in registry.")
        return self[name](*args, **kwargs)