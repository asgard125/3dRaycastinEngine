class EngineException(Exception):
    def __init__(self):
        pass

    def __str__(self):
        pass


class RayException(EngineException):
    def __init__(self):
        pass

    def __str__(self):
        pass


class EntityException(EngineException):
    def __init__(self):
        pass

    def __str__(self):
        pass


class IdentifierException(EngineException):
    def __init__(self):
        pass

    def __str__(self):
        pass


class EntitiesListException(EngineException):
    def __init__(self):
        pass

    def __str__(self):
        pass


class ObjectException(EngineException):
    def __init__(self):
        pass

    def __str__(self):
        pass


class CameraException(EngineException):
    def __init__(self):
        pass

    def __str__(self):
        pass


class EventSystemException(EngineException):
    def __init__(self):
        pass

    def __str__(self):
        pass


class GameException(EngineException):
    def __init__(self):
        pass

    def __str__(self):
        pass


class EntityReservedPropertyException(EntityException):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = "This property name is reserved"

    def __str__(self):
        return self.message


class EntityNonExistingPropertyException(EntityException):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = "This property doesn't exist"

    def __str__(self):
        return self.message


class EntitiesListEntityInListException(EntitiesListException):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = "This entity is already in this entity list"

    def __str__(self):
        return self.message


class EntitiesListEntityNotInListException(EntitiesListException):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = "This entity not in this entities list"

    def __str__(self):
        return self.message


class EventSystemEventNameNotInDictException(EventSystemException):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = "This entity not in this entities list"

    def __str__(self):
        return self.message


class EventSystemEventExistingNameException(EventSystemException):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = "This entity not in this entities list"

    def __str__(self):
        return self.message


class EventSystemEventNotInListException(EventSystemException):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = "This entity not in this entities list"

    def __str__(self):
        return self.message