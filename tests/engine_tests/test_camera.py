from lib.engine.engine import Entity, EntitiesList, HyperPlane, Ray, Camera, Object
from lib.math.base_classes import Point, CoordinateSystem, Vector


# Работоспособность функции получения всех лучей
# ------------------------------------------------------------
def test_camera_all_rays_creation():
    initial_pt_basis = Point([0, 0, 0])
    basis = ([1, 0, 0], [0, 1, 0], [0, 0, 1])
    cs = CoordinateSystem(basis, initial_pt_basis)
    initial_pt_camera = Point([0, 1, 0])
    direction_camera = Vector([1, 0, 0])
    initial_pt_hyperplane = Point([5, 1, 0])
    normal_hyperplane = Vector([1, 0, 0])
    hyperplane = HyperPlane(cs, initial_pt_hyperplane, normal_hyperplane)
    camera = Camera(cs, initial_pt_camera, direction_camera, 100, 60)
    assert camera.get_rays_matrix(50, 100)