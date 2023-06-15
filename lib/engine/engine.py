import copy
import math
import random
from lib.exceptions import engine_exceptions as ee
from lib.math import base_classes as bc
import json
import os


class Ray:
    """
        A class that implements a mathematical object ray and operations on them.
    """

    def __init__(self, cs: bc.CoordinateSystem, initial_pt: bc.Point, direction: bc.Vector):
        """Possible initialization options.

            Ray(cs: CoordinateSystem, initial_pt: Point, direction: Vector).
            Creates a ray with the specified coordinate system, initial point and direction.
        """
        self.cs = cs
        self.initial_pt = initial_pt
        self.direction = direction


class Identifier:
    """
        A class that implements an unique identifier.
    """
    identifiers = set()

    def __init__(self):
        """Possible initialization options.

            Identifier().
            Creates a unique identifier with generated value and adds it in all identifiers set.
        """
        self.value = self.__generate__()
        self.identifiers.add(self)

    @staticmethod
    def __generate__() -> int | float | str:
        """
        Method for creating unique identifier value
        Args:

        Raises:


        Returns:
            unique value which is not in set of all identifiers
        """
        value = random.randint(1, 1000)
        while value in [i.get_value() for i in list(Identifier.identifiers)]:
            value = random.randint(1, 1000)
        return value

    def get_value(self) -> int | float | str:
        """
        Method for returning value of an identifier
        Args:

        Raises:


        Returns:
            a value of identifier
        """
        return self.value


class Entity:
    """
        A class that implements an engine entity with the specified properties.
    """

    def __init__(self, cs: bc.CoordinateSystem):
        """Possible initialization options.

            Entity(cs: CoordinateSystem).
            Creates an Entity with the specified coordinate system, empty properties (as dictionary)
            and unique identifier as an object of class Identifier.
        """
        self.cs = cs
        self.properties = {}
        self.identifier = Identifier()

    def set_property(self, prop: str, value) -> None:
        """
        Method for setting new property of entity or changing existing property
        Args:
            prop (str): a name of a property
            value (any): a value of a property
        Raises:
            EntityReservedPropertyException
        Returns:

        """
        if prop in ['cs', 'properties', 'identifier']:
            raise ee.EntityReservedPropertyException
        self.properties[prop] = value

    def get_property(self, prop):
        """
        Method for getting a value of a property
        Args:
            prop (str): a name of a property

        Raises:
            EntityNonExistingPropertyException
        Returns:
            A property value if property exists
        """
        if prop not in ['cs', 'properties', 'identifier']:
            if prop not in self.properties:
                raise ee.EntityNonExistingPropertyException
            return copy.deepcopy(self.properties[prop])

    def remove_property(self, prop: str) -> None:
        """
        Method for removing existing property
        Args:
            prop (str): a name of a property

        Raises:
            EntityNonExistingPropertyException
            EntityReservedPropertyException
        Returns:

        """
        if prop in ['cs', 'properties', 'identifier']:
            raise ee.EntityReservedPropertyException
        if prop not in self.properties:
            raise ee.EntityNonExistingPropertyException
        del self.properties[prop]

    def __getitem__(self, prop: str):
        """
            Method for getting a value of property by accessing as index
            Args:
                prop (str): a name of a property

            Raises:
                EntityNonExistingPropertyException

            Returns:

        """
        return self.get_property(prop)

    def __setitem__(self, prop: str, value):
        """
            Method for setting new property of entity or changing existing property by accessing as index
            Args:
                prop (str): a name of a property
                value (any): a value of a property
            Raises:
                EntityReservedPropertyException
            Returns:

        """
        self.set_property(prop, value)

    def __delitem__(self, prop: str):
        """
            Method for removing existing property by accessing as index
            Args:
                prop (str): a name of a property

            Raises:
                EntityNonExistingPropertyException
                EntityReservedPropertyException
            Returns:

        """
        self.remove_property(prop)

    def __get__(self, prop):
        """
            Method for getting a value of property by accessing as attribute
            Args:
                prop (str): a name of a property

            Raises:
                EntityNonExistingPropertyException
                AttributeError

            Returns:

        """
        self.get_property(prop)

    def __set__(self, prop: str, value):
        """
            Method for setting new property of entity or changing existing property by accessing as attribute
            Args:
                prop (str): a name of a property
                value (any): a value of a property
            Raises:
                EntityReservedPropertyException
                AttributeError

            Returns:

        """
        self.set_property(prop, value)

    def __delete__(self, prop: str):
        """
            Method for removing existing property by accessing as attribute
            Args:
                prop (str): a name of a property

            Raises:
                EntityNonExistingPropertyException
                EntityReservedPropertyException
                AttributeError
            Returns:

        """
        self.remove_property(prop)


class Object(Entity):
    """
        A class that implements a some special case of engine entity with the specified properties and mathematical
        functions on it.
    """

    def __init__(self, cs: bc.CoordinateSystem, position: bc.Point, direction: bc.Vector):
        """Possible initialization options.

            Object(cs: bc.CoordinateSystem, position: bc.Point, direction: bc.Vector).
            Creates an Object with the specified coordinate system, unique identifier and position, direction as
            properties.
        """
        super().__init__(cs)
        self.set_property('position', position)
        self.set_property('direction', direction.normalize())

    def move(self, direction: bc.Vector, step) -> None:
        """
           Method for moving an object in the direction
            of a vector.
           Args:
               direction (Vector): a Vector of direction to move
               step: size of step of move
           Raises:

           Returns:

        """
        self.set_property('position', self.get_property('position') + (direction.normalize() * step))

    def planar_rotate(self, ind1: int, ind2: int, angle: float) -> None:
        """
           Method to rotate an object in a plane defined by two axes by a given angle.
           Args:
               inds (tuple(int, int)): an axes
               angle (float): an angle

           Raises:

           Returns:

        """
        self.set_direction(bc.Vector.to_vector(bc.Matrix.get_rotation_matrix(math.radians(angle), 3, ind1, ind2)
                                               * self.get_property('direction')))

    def rotate_3d(self, angle1: float, angle2: float, angle3: float) -> None:
        """
           Method to rotate a three-dimensional object at specified angles
           Args:
               angle1 (float): angle of rotating by x-axis
               angle2 (float): angle of rotating by y-axis
               angle3 (float): angle of rotating by z-axis

           Raises:

           Returns:

        """
        self.set_direction(bc.Vector.to_vector(bc.Matrix.get_teit_bryan_matrix(math.radians(angle1),
                                                                               math.radians(angle2),
                                                                               math.radians(angle3)) *
                                               self.get_property('direction')))

    def set_position(self, position: bc.Point) -> None:
        """
           A method to establish the new position of an object in space.
           Args:
               position (Point): a Point of new position

           Raises:

           Returns:

        """
        self.set_property('position', position)

    def set_direction(self, direction: bc.Vector) -> None:
        """
           Method to set the new direction
            of the "view" of an object in space.
           Args:
               direction (Vectors): a Vector of new direction.

           Raises:

           Returns:

        """
        self.set_property('direction', direction.normalize())

    def intersection_distance(self, ray: Ray) -> float:
        """
           method of finding the distance from the beginning of the ray to the point of intersection of the
           ray with the object. In the global class, it is considered an empty method that returns by
           default.
           Args:
               ray (Ray): a Ray class object

           Raises:

           Returns:
               -1 which means doesn't intersect

        """
        return -1


class Camera(Object):
    """
        A class that implements a some special case of engine object with additional initialization options.
    """
    DEFAULT_HORIZONTAL_RATIO = 16
    DEFAULT_VERTICAL_RATIO = 9

    def __init__(self, cs: bc.CoordinateSystem, position: bc.Point, direction: bc.Vector,
                 draw_distance: float, fov: float, vfov: float = None, look_at: bc.Point = None):
        """Possible initialization options.

            Camera(cs: CoordinateSystem, position: Point, direction: Vector, fov: float, draw_distance: float)
            Camera(cs: CoordinateSystem, position: Point, direction: Vector,
            fov: float, vfov: float, draw_distance: float)

            Camera(cs: CoordinateSystem, position: Point, direction: Vector,
            fov: float, look_at: Point, draw_distance: float)

            Camera(cs: CoordinateSystem, position: Point, direction: Vector,
            fov: float, vfov: float, look_at: Point, draw_distance: float)

            Creates a Camera with the specified coordinate system, and entities list in it.
            if vfov is not specified it calculates automatically
            as fov / DEFAULT_HORIZONTAL_RATIO * DEFAULT_VERTICAL_RATIO.
        """
        super().__init__(cs, position, direction)
        self.set_property('draw_distance', draw_distance)
        self.set_property('fov', fov)
        self.set_property('vfov', vfov)
        self.set_property('look_at', look_at)
        self.set_property("cached_rays", None)
        self.set_property("resolution_width", None)
        self.set_property("resolution_height", None)

        self.set_property("init_direction", bc.Vector([1, 0, 0]))

        self.rotation_angle_horizontal = 0
        self.set_property("rotation_angle_horizontal", 0)
        self.set_property("rotation_angle_vertical", 0)

        if look_at is not None:
            new_direction = bc.Vector.to_vector(position - bc.Vector.to_vector(look_at))
            self.set_property('direction', new_direction.normalize())

    def init_rays(self, n: int, m: int) -> None:
        """
           A method that generates initial rays with specified fov, width and height of screen.
           Initial rays are rays for given point and vector [1, 0, 0].
           Args:
               n (int): vertical size of screen.
               m (int): horizontal size of screen.

           Raises:

           Returns:

        """
        direction = self.get_property("init_direction")
        fov = self.get_property("fov")
        if self.get_property("vfov") is None:
            vfov = round(2 * math.degrees(math.atan((0.5 * n) /
                                                    (0.5 * m / math.tan(math.radians(fov / 2))))))
            self.set_property("vfov", vfov)
        else:
            vfov = self.get_property("vfov")

        vertical_angles = [vfov / n * i - vfov / 2 for i in range(n)]
        horizontal_angles = [fov / m * j - fov / 2 for j in range(m)]
        all_rays_data = [[bc.Vector.to_vector((bc.Matrix.get_rotation_matrix(math.radians(horizontal_angles[j]), 3, 0, 2) *
                                               (bc.Matrix.get_rotation_matrix(math.radians(vertical_angles[i]), 3, 0, 1) *
                                                direction))) for j in range(m)] for i in range(n)]

        # # fisheye fix
        # for i in range(n):
        #     for j in range(m):
        #         all_rays_data[i][j] = all_rays_data[i][j] * (abs(self.get_property("init_direction"))
        #                                                      * abs(self.get_property("init_direction")) /
        #                                                      (self.get_property("init_direction") & all_rays_data[i][j]))


        self.set_property("resolution_width", m)
        self.set_property("resolution_height", n)
        self.set_property("initial_rays_vectors", all_rays_data)

    def get_rays_matrix(self) -> list:
        """
           A method that returns rays rotated by camera's horizontal and vertical rotated angles.
           Args:


           Raises:

           Returns:
            Ray object class with camera's point, coordinate system and direction from initial rays, rotated by angles.
        """
        height = self.get_property("resolution_height")
        width = self.get_property("resolution_width")
        camera_position = self.get_property("position")
        initial_rays = self.get_property("initial_rays_vectors")

        vertical_rotation_matrix = bc.Matrix.get_rotation_matrix(
            math.radians(self.get_property("rotation_angle_vertical")), 3, 0, 1)
        horizontal_rotation_matrix = bc.Matrix.get_rotation_matrix(
            math.radians(self.get_property("rotation_angle_horizontal")), 3, 0, 2)
        current_rays_vectors = [[bc.Vector.to_vector(horizontal_rotation_matrix * (vertical_rotation_matrix *
                         initial_rays[i][j])) for j in range(width)] for i in range(height)]

        current_rays = []
        for i in range(height):
            rays_line = []
            for j in range(width):
                current_ray = Ray(self.cs, camera_position, current_rays_vectors[i][j])
                rays_line.append(current_ray)
            current_rays.append(rays_line)
        return current_rays

    def rotate_camera_view(self, angle: float, rotation_type: str) -> None:
        """
           A method that sets new rotation camera angles and rotates direction of camera.
           Args:


           Raises:

           Returns:

        """
        if rotation_type == "vertical":
            if angle < 0:
                self.set_property("rotation_angle_vertical", max(-89 + self.get_property("vfov") / 2,
                                                                 self.get_property("rotation_angle_vertical") + angle))
            else:
                self.set_property("rotation_angle_vertical", min(89 - self.get_property("vfov") / 2,
                                                                 self.get_property(
                                                                     "rotation_angle_vertical") + angle))
        elif rotation_type == "horizontal":
            self.set_property("rotation_angle_horizontal", (self.get_property("rotation_angle_horizontal")
                                                            + angle) % 360)
        horizontal_rotation_matrix = bc.Matrix.get_rotation_matrix(
            math.radians(self.get_property("rotation_angle_horizontal")), 3, 0, 2)
        vertical_rotation_matrix = bc.Matrix.get_rotation_matrix(
            math.radians(self.get_property("rotation_angle_vertical")), 3, 0, 1)

        self.set_property("direction", bc.Vector.to_vector(horizontal_rotation_matrix * (vertical_rotation_matrix *
                                                            self.get_property("init_direction"))).normalize())


class HyperPlane(Object):
    """
        A class that implements a hyperplane object in a given coordinate system.
    """

    def __init__(self, cs: bc.CoordinateSystem, position: bc.Point, normal: bc.Vector):
        """Possible initialization options.

            HyperPlane(cs: bc.CoordinateSystem, position: bc.Point, normal: bc.Vector).
            Creates an HyperPlane object with the specified coordinate system, unique identifier and position, normal as
            properties.
        """
        super().__init__(cs, position, normal)
        self.set_property('normal', normal.normalize())

    def planar_rotate(self, ind1, ind2, angle: float) -> None:
        """
           Method to rotate a hyperplane normal in space.
           Args:
               inds (tuple(int, int)): an axes
               angle (float): an angle

           Raises:

           Returns:

        """
        self.set_property('normal', bc.Vector.to_vector(bc.Matrix.get_rotation_matrix(math.radians(angle),
                                                                                      3, ind1, ind2)
                                                        * self.get_property('normal')).normalize())

    def rotate_3d(self, angle1: float, angle2: float, angle3: float) -> None:
        """
           Method to rotate a three-dimensional plane at specified angles
           Args:
               angle1 (float): angle of rotating by x-axis
               angle2 (float): angle of rotating by y-axis
               angle3 (float): angle of rotating by z-axis

           Raises:

           Returns:

        """
        new_normal_vector = bc.Vector.to_vector(
            bc.Matrix.get_teit_bryan_matrix(math.radians(angle1), math.radians(angle2), math.radians(angle3))
            * self.get_property('normal'))
        self.set_property('normal', new_normal_vector.normalize())

    def intersection_distance(self, ray: Ray) -> float:
        """
           method of finding the distance from the beginning of the ray to the point of intersection of the
           ray with the hyperplane.
           Args:
               ray (Ray): a Ray class object.

           Raises:

           Returns:
               the shortest distance from start point of the ray to hyperplane.

        """


        a = self.get_property('normal') & bc.Vector.to_vector(
        ray.initial_pt - bc.Vector.to_vector(self.get_property('position')))
        b = self.get_property('normal') & ray.direction
        if a == 0:
            return 0
        elif b == 0:
            return -1
        else:
            t = -a / b
            if t < 0:
                return -1
            else:
                return t


class HyperElipsoid(Object):
    """
        A class that implements a hyperelipsoid object in a given coordinate system.
    """

    def __init__(self, cs: bc.CoordinateSystem, position: bc.Point, semiaxes: list[float]):
        """Possible initialization options.

            HyperElipsoid(cs: bc.CoordinateSystem, position: bc.Point, semiaxes: list[float]).
            Creates an HyperPlane object with the specified coordinate system, unique identifier and position, direction,
            semiaxes as properties.
        """
        super().__init__(cs, position, bc.Vector([0, 0, 0]))
        self.set_property('semiaxes', semiaxes)

    def planar_rotate(self, ind1, ind2, angle: float) -> None:
        """
           Method to rotate a hyperelipsoid in space.
           Args:
               inds (tuple(int, int)): an axes
               angle (float): an angle

           Raises:

           Returns:

        """
        self.cs = bc.CoordinateSystem((
            [bc.Vector.to_vector(bc.Matrix.get_rotation_matrix(math.radians(angle),
                                                               self.cs.get_dim_size(), ind1, ind2) *
                                 vec) for vec in self.cs.basis.basis]),
            bc.Point.to_point(bc.Matrix.get_rotation_matrix(math.radians(angle), self.cs.get_dim_size(),
                                                            ind1, ind2)
                              * self.cs.initial_pt))
        self.set_position(bc.Point.to_point(self.cs.basis.basis_decompose(self.get_property('position'))))

    def rotate_3d(self, angle1: float, angle2: float, angle3: float) -> None:
        """
           Method to rotate a three-dimensional elipsoid at specified angles
           Args:
               angle1 (float): angle of rotating by x-axis
               angle2 (float): angle of rotating by y-axis
               angle3 (float): angle of rotating by z-axis

           Raises:

           Returns:

        """
        self.cs = bc.CoordinateSystem(([bc.Vector.to_vector(bc.Matrix.get_teit_bryan_matrix(math.radians(angle1),
                                                                                            math.radians(angle2),
                                                                                            math.radians(angle3)) *
                                                            vec) for vec in self.cs.basis.basis]),
                                      bc.Point.to_point(bc.Matrix.get_teit_bryan_matrix(math.radians(angle1),
                                                                                        math.radians(angle2),
                                                                                        math.radians(angle3))
                                                        * self.cs.initial_pt))
        self.set_position(bc.Point.to_point(self.cs.basis.basis_decompose(self.get_property('position'))))

    def intersection_distance(self, ray: Ray) -> float:
        """
           method of finding the distance from the beginning of the ray to the point of intersection of the
           ray with the hyperelipsoid.
           Args:
               ray (Ray): a Ray class object.

           Raises:

           Returns:
               the shortest distance from start point of the ray to hyperelipsoid.

        """
        ray_in_elipsoid_basis = Ray(self.cs, bc.Point.to_point(self.cs.basis.basis_decompose(ray.initial_pt)),
                                    self.cs.basis.basis_decompose(ray.direction))

        dist_from_centre = [ray_in_elipsoid_basis.initial_pt[i] - self.get_property('position')[i]
                            for i in range(self.cs.get_dim_size())]
        t_a = 0
        t_b = 0
        t_c = 0
        for i in range(self.cs.get_dim_size()):
            semiaxes_multiplier = 1
            for j in range(self.cs.get_dim_size()):
                if i != j:
                    semiaxes_multiplier *= (self.get_property('semiaxes')[j] ** 2)
            t_a += (ray_in_elipsoid_basis.direction[i] ** 2 * semiaxes_multiplier)
            t_b += (2 * ray_in_elipsoid_basis.direction[i] * semiaxes_multiplier * dist_from_centre[i])
            t_c += dist_from_centre[i] ** 2 * semiaxes_multiplier
        semiaxes_squares_multiplication = 1
        for i in range(self.cs.get_dim_size()):
            semiaxes_squares_multiplication *= (self.get_property('semiaxes')[i] ** 2)
        t_c -= semiaxes_squares_multiplication
        t = -1
        if t_a == 0:
            if t_b == 0:
                return -1
            else:
                t = -t_c / t_b
        else:
            D = t_b ** 2 - 4 * t_a * t_c
            if D < 0:
                return -1
            elif D == 0:
                t = -t_b / 2 / t_a
            else:
                t = min((-t_b + math.sqrt(D)) / 2 / t_a, (-t_b - math.sqrt(D)) / 2 / t_a)
        if t < 0:
            return -1
        else:
            return t


class EntitiesList:
    """
        A class that implements an engine entity storage.
    """

    def __init__(self, *entities: Entity | Object):
        """Possible initialization options.

            EntitiesList(Entity, ...).
            Creates an EntitiesList with the specified entities in it.
        """
        self.entities = list(entities)

    def append(self, entity: Entity) -> None:
        """
           Method adding entity to the list
           Args:
               entity (Entity): an entity object not in this list

           Raises:
               EntitiesListEntityInListException

           Returns:

        """
        if self.get(entity.identifier) is not None:
            raise ee.EntitiesListEntityInListException
        self.entities.append(entity)

    def remove(self, id: Identifier) -> None:
        """
           Method for removing entity from the list
           Args:
               id (Identifier): an Identifier object of entity

           Raises:
               EntitiesListEntityNotInListException

           Returns:

        """
        if self.get(id) is None:
            raise ee.EntitiesListEntityNotInListException
        self.entities.remove(self.get(id))

    def get(self, id: Identifier) -> Entity | None:
        """
           Method for getting entity by its identifier
           Args:
               id (Identifier): an Identifier object of entity

           Raises:

           Returns:
            Entity object with matching Identifier or None
        """
        for i in range(len(self.entities)):
            if self.entities[i].identifier == id:
                return self.entities[i]
        return None

    def length(self):
        return len(self.entities)

    def get_by_index(self, ind: int) -> Entity | None:
        """
           Method for getting entity by its identifier
           Args:
               ind (int): index in list of entities

           Raises:

           Returns:
            Entity object with matching Identifier or None
        """
        return self.entities[ind]

    def exec(self, func) -> None:
        """
           Method for applying the function of one argument (entity) to all entities of this
                list.
           Args:
               func (function): a function of one argument (entity)

           Raises:

           Returns:

        """
        for i in range(len(self.entities)):
            func(self.entities[i])

    def __getitem__(self, id: Identifier):
        """
           Method for getting entity by its identifier as index
           Args:
               id (Identifier): an Identifier object of entity

           Raises:

           Returns:
                Entity object with matching Identifier or None
        """
        return self.get(id)


class Canvas:
    """
    A class that implements basic canvas for rendering.
    Already holds all the information about the game: list of entities, a system coordinates.
    """

    def __init__(self, height: int, width: int, cs: bc.CoordinateSystem, entities: EntitiesList):
        """Possible initialization options.

            Canvas(n: int, m: int, cs: CoordinateSystem, entities: EntitiesList, distances: Matrix).
            Creates a Canvas with the specified screen size, coordinate system, entities and distances in it.
        """
        self.height = height
        self.width = width
        self.cs = cs
        self.entities = entities
        self.distances = dict()

    def draw(self) -> None:
        """
        A method that renders a known distances matrix in the Game class.
        Args:

        Raises:

        Returns:

        """
        pass

    def update(self, camera: Camera) -> None:
        """
        A method that updates
        the distances matrix according to the view from the camera. Performs a full run
        through all entities, on the basis of which, using the intersection_distance method,
        builds a new distances matrix.
        Args:
            camera (Camera):
        Raises:

        Returns:

        """
        all_rays_matrix = camera.get_rays_matrix()
        new_distances = []
        for i in range(self.height):
            distances_line = []
            for j in range(self.width):
                best_dist = -1
                for entity_ind in range(self.entities.length()):
                    if isinstance(self.entities.get_by_index(entity_ind), Object):
                        dist = self.entities.get_by_index(entity_ind).intersection_distance(all_rays_matrix[i][j])
                        if best_dist == -1:
                            best_dist = dist
                        else:
                            if dist > -1:
                                best_dist = min(dist, best_dist)
                distances_line.append(best_dist)
            new_distances.append(distances_line)
        self.distances["distances"] = new_distances
        self.distances["draw_distance"] = camera.get_property("draw_distance")


class Console(Canvas):
    """
    A class for rendering an image in the console.
    """

    def __init__(self, height: int, width: int, cs: bc.CoordinateSystem, entities: EntitiesList,
                 charmap: str = None):
        """Possible initialization options.

            Console(n: int, m: int, cs: CoordinateSystem, entities: EntitiesList, distances: Matrix, charmap: str = None).
            Creates a Console with the specified screen size, coordinate system, entities, distances and charmap in it.
            If charmap is None, uses standard specified symbol string.
        """
        super().__init__(height, width, cs, entities)
        self.charmap = charmap

    def draw(self) -> None:
        """
            A method that renders a known distances matrix in the Game class to the console
            using the rendering library and symbols specified in this class.
            Args:

            Raises:

            Returns:

        """
        clear = lambda: os.system('cls')
        clear()
        print('-' * (self.width * 3 + 2))
        if self.distances is None:
            for n in range(self.height):
                print("|" + (" " * self.width * 3) + "|")
        else:
            for n in range(self.height - 1, -1, -1):
                print("|", end="")
                for m in range(self.width):
                    if (self.distances["distances"][n][m] == -1 or
                            round(self.distances["distances"][n][m], 1) >
                            self.distances["draw_distance"]):
                        print(" " * 3, end="")
                    else:
                        charmap_index = round((1 - round(self.distances["distances"][n][m], 1) / self.distances["draw_distance"]) * \
                                        (len(self.charmap) - 1))
                        print(self.charmap[charmap_index] * 3, end="")
                print("|", end="")
                print()
        print('-' * (self.width * 3 + 2))


class EventSystem:
    """
    A class responsible for implementing and handling the event. It contains information
    about existing events in the game.
    """

    def __init__(self, events: dict[str: list[callable]]):
        """Possible initialization options.

            EventSystem(events: dict[str: list[callable]]): creates an object of class with dict of list of functions by
            name.
        """
        self.events = events

    def add(self, name: str) -> None:
        """
            A method that adds a new name for an event to the event system.
            Args:
                name (str): a unique name of event list
            Raises:
                EventSystemEventExistingNameException
            Returns:

        """
        if name in self.events:
            raise ee.EventSystemEventExistingNameException('This events name already in dict')
        self.events[name] = []

    def remove(self, name: str) -> None:
        """
            A method to remove events from the system by a given name.
            Args:
                name (str): a unique name of event list
            Raises:
                EventSystemEventNameNotInDictException
            Returns:

        """
        if name not in self.events:
            raise ee.EventSystemEventNameNotInDictException('This event name is not in dict')
        del self.events[name]

    def handle(self, name: str, function: callable) -> None:
        """
            A method that adds a handler function to an array of functions for a given event name.
            Args:
                name (str): a unique name of event list
                function (callable): a function which is a game event
            Raises:
                EventSystemEventNameNotInDictException
            Returns:

        """
        if name not in self.events:
            raise ee.EventSystemEventNameNotInDictException('This event name is not in dict')
        self.events[name].append(function)

    def remove_handled(self, name, function: callable) -> None:
        """
            A method that removes a handler function from an array of functions for a given event name.
            Args:
                name (str): a unique name of event list
                function (callable): a function which is an event
            Raises:
                EventSystemEventNameNotInDictException
                EventSystemEventNotInListException
            Returns:

        """
        if name not in self.events:
            raise ee.EventSystemEventNameNotInDictException('This event name is not in dict')
        if function not in self.events[name]:
            raise ee.EventSystemEventNotInListException('This event not in this named list')
        self.events[name].remove(function)

    def trigger(self, name: str, *args) -> None:
        """
            A method that calls all handler functions in the event space for
            a given event name with the arguments passed.
            Args:
                name (str): a unique name of event list
                args (): function args
            Raises:
                EventSystemEventNameNotInDictException
            Returns:

        """
        if name not in self.events:
            raise ee.EventSystemEventNameNotInDictException('This event name is not in dict')
        for event in self.events[name]:
            event(*args)

    def get_handled(self, name: str) -> list:
        """
            A method that returns all handler functions for a given event.
            Args:
                name (str): a unique name of event list

            Raises:
                EventSystemEventNameNotInDictException
            Returns:
                list of events by name
        """
        if name not in self.events:
            raise ee.EventSystemEventNameNotInDictException('This event name is not in dict')
        return self.events[name]

    def __getitem__(self, name: str):
        """
            A method that returns all handler functions for a given event as index.
            Args:
                name (str): a unique name of event list

            Raises:
                EventSystemEventNameNotInDictException
            Returns:
                list of events by name
        """
        return self.get_handled(name)


class Configuration:
    """
        General configuration class of the game.
    """

    def __init__(self, filepath: str | None):
        """Possible initialization options.

            Configuration(filepath: str): creates a configuration class with variables from filepath. If file doesn't
            exist it uses default variables.
        """
        self.filepath = filepath
        self.configuration = dict()
        if filepath is None:
            self.execute_file("config/default.json", 'rewrite')
        else:
            self.execute_file(filepath, 'rewrite')

    def set_variable(self, var: str, value: any) -> None:
        """
            A method that returns all handler functions for a given event as index.
            Args:
                var (str): a name of variable.
                value (ant): a value for variable. If None deletes the variable name from configuration
            Raises:

            Returns:

        """
        self.configuration[var] = value

    def get_variable(self, var: str) -> any:
        """
            A method that returns the value of the variable in the current configuration.
            Args:
                var (str): a name of variable.

            Raises:

            Returns:
                value from configuration variables by variable name
        """
        return self.configuration[var]

    def execute_file(self, filepath: str, operation_type="update") -> None:
        """
            A method that updates the values of the values loaded from the file.
            Args:
                filepath (str): a path to file.
                operation_type (str): type of operation. if update - adds variables to configuration or update values
                of existing variables. If rewrite replaces old configuration by new from file.
            Raises:

            Returns:

        """
        with open(filepath) as json_file:
            new_configuration = json.load(json_file)
        if operation_type == "rewrite":
            self.configuration = new_configuration
        elif operation_type == "update":
            for key in new_configuration:
                self.set_variable(key, new_configuration[key])

    def save(self, filepath: str = None) -> None:
        """
            A method that saves the configuration to a file.
            If the file is not specified, saves it to a file with
            the file name from the filepath field of the current configuration.
            Args:
                filepath (str): a path to file.

            Raises:

            Returns:

        """
        if filepath is None:
            filepath = self.filepath
        with open(filepath, 'w') as f:
            json.dump(self.configuration, f)

    def __getitem__(self, var: str):
        """
            A method that returns the value of the variable in the current configuration as index.
            Args:
                var (str): a unique name of game variable

            Raises:

            Returns:
                value from configuration variables by variable name
        """
        return self.get_variable(var)


class Game:
    """
        A class that implements a main engine class which has methods to make operations in real time.
    """

    def __init__(self, cs: bc.CoordinateSystem = None, entities: EntitiesList = None, es: EventSystem = None,
                 config: Configuration = None):
        """Possible initialization options.

            Game(cs: CoordinateSystem, entities: EntitiesList, es: EventSystem).
            Creates a Game with the specified coordinate system, entities and event system list in it.
        """
        self.cs = cs
        self.entities = entities
        self.es = es
        self.main_camera = None
        self.game_config = config

        self.game_entity_class = None
        self.game_ray_class = None
        self.game_camera_class = None
        self.game_object_class = None
        self.game_hyperplane_class = None
        self.game_hyperelipsoid_class = None
        self.game_canvas_class = None
        self.game_console_class = None

    def init_by_config(self) -> None:
        """
            Method to initialize a game by data from game config.
            Args:

            Raises:

            Returns:

        """
        # инициализация координатной системы
        game_basis = [bc.Vector(vec) for vec in self.game_config.get_variable("coordinate_system")["basis"]]
        initial_point = bc.Point(self.game_config.get_variable("coordinate_system")["initial_point"])
        self.cs = bc.CoordinateSystem(game_basis, initial_point)

        # создание главной камеры
        camera_position = bc.Point(self.game_config.get_variable("camera")["position"])
        camera_direction = bc.Vector([1, 0, 0])
        camera_fov = self.game_config.get_variable("camera")["fov"]
        camera_vfov = self.game_config.get_variable("camera")["vfov"]
        camera_draw_distance = self.game_config.get_variable("camera")["draw_distance"]
        game_camera = self.get_camera_class()
        self.main_camera = game_camera(camera_position, camera_direction, camera_draw_distance, camera_fov, camera_vfov,
                                       None)

        self.main_camera.init_rays(self.game_config.get_variable("video_data")["height"],
                                   self.game_config.get_variable("video_data")["width"])

        # инициализация сущностей
        self.entities = EntitiesList()
        for entitiy_data in self.game_config.get_variable("entities"):
            if entitiy_data["type"] == "hyperplane":
                game_hyperplane = self.get_hyperplane_class()
                hyperplane_normal = bc.Vector(entitiy_data["normal"])
                hyperplane_position = bc.Point(entitiy_data["position"])
                self.entities.append(game_hyperplane(hyperplane_position, hyperplane_normal))
            elif entitiy_data["type"] == "hyperelipsoid":
                game_hyperpelipsoid = self.get_hyperelipsoid_class()
                hyperelipsoid_position = bc.Point(entitiy_data["position"])
                hyperelipsoid_semiaxes = entitiy_data["semiaxes"]
                self.entities.append(game_hyperpelipsoid(hyperelipsoid_position, hyperelipsoid_semiaxes))
            elif entitiy_data["type"] == "object":
                game_object = self.get_object_class()
                object_position = bc.Point(entitiy_data["position"])
                object_direction = entitiy_data["direction"]
                self.entities.append(game_object(object_position, object_direction))
            elif entitiy_data["type"] == "entity":
                game_entity = self.get_entity_class()()
                for prop in entitiy_data["properties"]:
                    game_entity.set_property(prop["name"], prop["value"])
                self.entities.append(game_entity)

        # система эвентов
        if self.es is None:
            events = dict()
            self.es = EventSystem(events)

    def run(self) -> None:
        """
            Method executing the game launch script.
            Args:

            Raises:

            Returns:

        """
        pass

    def update(self) -> None:
        """
            Method that executes the script for updating the game process (updating information).
            Args:

            Raises:

            Returns:

        """
        pass

    def exit(self) -> None:
        """
            Method that executes the exit script from the game process.
            Args:

            Raises:

            Returns:

        """
        pass

    def get_entity_class(self):
        """
        Method for creating an object of Entity class with a coordinate system from the Game class.
        Args:

        Raises:

        Returns:
            GameEntity class
        """
        if self.game_entity_class is not None:
            return self.game_entity_class

        class GameEntity(Entity):
            def __init__(pself):
                super().__init__(self.cs)

        self.game_entity_class = GameEntity
        return self.game_entity_class

    def get_ray_class(self):
        """
        Method for creating an object of Ray class with a coordinate system from the Game class.
        Args:

        Raises:

        Returns:
            GameRay class
        """
        if self.game_ray_class is not None:
            return self.game_ray_class

        class GameRay(Ray):
            def __init__(pself, initial_pt: bc.Point, direction: bc.Vector):
                super().__init__(self.cs, initial_pt, direction)

        self.game_ray_class = GameRay
        return self.game_ray_class

    def get_object_class(self):
        """
        Method for creating an object of Object class with a coordinate system from the Game class.
        Args:

        Raises:

        Returns:
            GameObject class
        """
        if self.game_object_class is not None:
            return self.game_object_class

        class GameObject(Object):
            def __init__(pself, position: bc.Point, direction: bc.Vector):
                super().__init__(self.cs, position, direction)

        self.game_object_class = GameObject
        return self.game_object_class

    def get_camera_class(self):
        """
        Method for creating an object of Camera class with a coordinate system from the Game class.
        Args:

        Raises:

        Returns:
            GameCamera class
        """
        if self.game_camera_class is not None:
            return self.game_camera_class

        class GameCamera(Camera):
            def __init__(pself, position: bc.Point, direction: bc.Vector,
                         draw_distance: float, fov: float, vfov: float = None, look_at: bc.Point = None):
                super().__init__(self.cs, position, direction, draw_distance, fov, vfov, look_at)

        self.game_camera_class = GameCamera
        return self.game_camera_class

    def get_hyperplane_class(self):
        """
        Method for creating an object of HyperPlane class with a coordinate system from the Game class.
        Args:

        Raises:

        Returns:
            GameHyperPlane class
        """
        if self.game_hyperplane_class is not None:
            return self.game_hyperplane_class

        class GameHyperPlane(HyperPlane):
            def __init__(pself, position, normal):
                super().__init__(self.cs, position, normal)

        self.game_hyperplane_class = GameHyperPlane
        return self.game_hyperplane_class

    def get_hyperelipsoid_class(self):
        """
        Method for creating an object of HyperElipsoid class with a coordinate system from the Game class.
        Args:

        Raises:

        Returns:
            GameHyperElipsoid class
        """
        if self.game_hyperelipsoid_class is not None:
            return self.game_hyperelipsoid_class

        class GameHyperElipsoid(HyperElipsoid):
            def __init__(pself, position, semiaxes):
                super().__init__(self.cs, position, semiaxes)

        self.game_hyperelipsoid_class = GameHyperElipsoid
        return self.game_hyperelipsoid_class

    def get_canvas_class(self):
        """
        Method for creating an object of GameCanvas class with a coordinate system and entities from the Game class.
        Args:

        Raises:

        Returns:
            GameCanvas class
        """
        if self.game_canvas_class is not None:
            return self.game_canvas_class

        class GameCanvas(Canvas):
            def __init__(pself, n, m, distances):
                super().__init__(n, m, self.cs, self.entities, distances)

        self.game_canvas_class = GameCanvas
        return self.game_canvas_class

    def get_event_system(self):
        """
        Method that returns an object of the event system class for this instance of the game.
        Args:

        Raises:

        Returns:
            game eventsystem class object
        """
        return self.es

    def get_console_class(self):
        """
        Method for creating an object of GameConsole class with a coordinate system and entities from the Game class.
        Args:

        Raises:

        Returns:
            GameConsole class
        """
        if self.game_console_class is not None:
            return self.game_console_class

        class GameConsole(Console):
            def __init__(pself, n: int, m: int, charmap: str):
                super().__init__(n, m, self.cs, self.entities, charmap)

        self.game_console_class = GameConsole
        return self.game_console_class


def move_camera_3d(object: Object, entities: EntitiesList, step: float, direction: str) -> None:
    direction_vector = None
    if direction == "forward":
        direction_vector = object.get_property('direction')
    elif direction == "back":
        direction_vector = bc.Vector.to_vector(bc.Matrix.get_rotation_matrix(math.radians(180), 3, 0, 2)
                                               * object.get_property('direction'))
    elif direction == "left":
        direction_vector = bc.Vector.to_vector(bc.Matrix.get_rotation_matrix(math.radians(270), 3, 0, 2)
                                               * object.get_property('direction'))
    elif direction == "right":
        direction_vector = bc.Vector.to_vector(bc.Matrix.get_rotation_matrix(math.radians(90), 3, 0, 2)
                                               * object.get_property('direction'))
    elif direction == "up":
        direction_vector = bc.Vector([i for i in object.cs.basis.basis[1]])
    elif direction == "down":
        direction_vector = bc.Vector([i for i in object.cs.basis.basis[1]]) * -1
    intersect_ray_checker = Ray(object.cs, object.get_property("position"), direction_vector)
    min_dist = -1
    for entity_ind in range(entities.length()):
        if isinstance(entities.get_by_index(entity_ind), Object):
            dist = entities.get_by_index(entity_ind).intersection_distance(intersect_ray_checker)
            if min_dist == -1:
                min_dist = dist
            else:
                if dist > -1:
                    min_dist = min(dist, min_dist)
    if min_dist > step or min_dist == -1:
        object.move(direction_vector, step)


def rotate_camera_3d(camera: Camera, angle: float, direction: str):
    if direction == "up":
        camera.rotate_camera_view(angle, "vertical")
    elif direction == "down":
        camera.rotate_camera_view(-angle, "vertical")
    if direction == "left":
        camera.rotate_camera_view(-angle, "horizontal")
    if direction == "right":
        camera.rotate_camera_view(angle, "horizontal")
