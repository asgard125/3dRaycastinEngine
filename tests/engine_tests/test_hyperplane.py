import math

from lib.engine.engine import Entity, EntitiesList, HyperPlane, Ray
from lib.math.base_classes import Point, CoordinateSystem, Vector


# Пересечение гиперплоскости и луча
# ------------------------------------------------------------
def test_hyperplane_intersection_distance_intersect():
    initial_pt_basis = Point([0, 0, 0])
    basis = (Vector([1, 0, 0]), Vector([0, 1, 0]), Vector([0, 0, 1]))
    cs = CoordinateSystem(basis, initial_pt_basis)
    initial_pt_ray = Point([0, 1, 0])
    direction_ray = Vector([1, 0, 0])
    initial_pt_hyperplane = Point([5, 1, 0])
    normal_hyperplane = Vector([1, 0, 0])
    ray = Ray(cs, initial_pt_ray, direction_ray)
    hyperplane = HyperPlane(cs, initial_pt_hyperplane, normal_hyperplane)
    res = hyperplane.intersection_distance(ray)
    assert res == 5


def test_hyperplane_intersection_distance_not_intersect():
    initial_pt_basis = Point([0, 0, 0])
    basis = (Vector([1, 0, 0]), Vector([0, 1, 0]), Vector([0, 0, 1]))
    cs = CoordinateSystem(basis, initial_pt_basis)
    initial_pt_ray = Point([0, 1, 0])
    direction_ray = Vector([-1, 1, -1])
    initial_pt_hyperplane = Point([1, 0, 0])
    normal_hyperplane = Vector([1, 0, 0])
    ray = Ray(cs, initial_pt_ray, direction_ray)
    hyperplane = HyperPlane(cs, initial_pt_hyperplane, normal_hyperplane)
    res = hyperplane.intersection_distance(ray)
    assert res == -1


def test_hyperplane_intersection_distance_intersect_2():
    initial_pt_basis = Point([0, 0, 0])
    basis = (Vector([1, 0, 0]), Vector([0, 1, 0]), Vector([0, 0, 1]))
    cs = CoordinateSystem(basis, initial_pt_basis)
    initial_pt_ray = Point([0, 1, 0])
    direction_ray = Vector([1, 0, 0])
    initial_pt_hyperplane = Point([-5, 1, 0])
    normal_hyperplane = Vector([1, 0, 0])
    ray = Ray(cs, initial_pt_ray, direction_ray)
    hyperplane = HyperPlane(cs, initial_pt_hyperplane, normal_hyperplane)
    res = hyperplane.intersection_distance(ray)
    assert res == -1


def test_hyperplane_intersection_distance_intersect_3():
    initial_pt_basis = Point([0, 0, 0])
    basis = (Vector([1, 0, 0]), Vector([0, 1, 0]), Vector([0, 0, 1]))
    cs = CoordinateSystem(basis, initial_pt_basis)
    initial_pt_ray = Point([0, 1, 0])
    direction_ray = Vector([0, -0.5, -1])
    initial_pt_hyperplane = Point([0, 0, 0])
    normal_hyperplane = Vector([0, 1, 0])
    ray = Ray(cs, initial_pt_ray, direction_ray)
    hyperplane = HyperPlane(cs, initial_pt_hyperplane, normal_hyperplane)
    res = hyperplane.intersection_distance(ray)
    assert res != -1


def test_hyperplane_intersection_distance_ray_in_plane():
    initial_pt_basis = Point([0, 0, 0])
    basis = (Vector([1, 0, 0]), Vector([0, 1, 0]), Vector([0, 0, 1]))
    cs = CoordinateSystem(basis, initial_pt_basis)
    initial_pt_ray = Point([0, 1, 0])
    direction_ray = Vector([0, -1, 0])
    initial_pt_hyperplane = Point([0, 1, 0])
    normal_hyperplane = Vector([0, 1, 0])
    ray = Ray(cs, initial_pt_ray, direction_ray)
    hyperplane = HyperPlane(cs, initial_pt_hyperplane, normal_hyperplane)
    res = hyperplane.intersection_distance(ray)
    assert res == 0