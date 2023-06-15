import math

from lib.engine.engine import HyperElipsoid, Ray
from lib.math.base_classes import Point, CoordinateSystem, Vector


# Пересечение гиперэлипсоида и луча
# ------------------------------------------------------------
def test_hyperplane_intersection_distance_intersect():
    initial_pt_basis = Point([0, 0, 0])
    basis = (Vector([1, 0, 0]), Vector([0, 1, 0]), Vector([0, 0, 1]))
    cs = CoordinateSystem(basis, initial_pt_basis)
    initial_pt_ray = Point([0, 1, 0])
    direction_ray = Vector([1, 0, 0])
    initial_pt_hyperelipsoid = Point([5, 1, 0])
    direction_hyperelipsoid = Vector([1, 0, 0])
    semiaxes = [4, 7, 4]
    ray = Ray(cs, initial_pt_ray, direction_ray)
    hyperplane = HyperElipsoid(cs, initial_pt_hyperelipsoid, semiaxes)
    res = hyperplane.intersection_distance(ray)
    print(res)
    assert res == 1


def test_hyperplane_intersection_distance_not_intersect():
    initial_pt_basis = Point([0, 0, 0])
    basis = (Vector([1, 0, 0]), Vector([0, 1, 0]), Vector([0, 0, 1]))
    cs = CoordinateSystem(basis, initial_pt_basis)
    initial_pt_ray = Point([-7, 1, 0])
    direction_ray = Vector([-1, 0, 0])
    initial_pt_hyperelipsoid = Point([5, 1, 0])
    direction_hyperelipsoid = Vector([1, 0, 0])
    semiaxes = [5, 6, 4]
    ray = Ray(cs, initial_pt_ray, direction_ray)
    hyperplane = HyperElipsoid(cs, initial_pt_hyperelipsoid, semiaxes)
    res = hyperplane.intersection_distance(ray)
    print(res)
    assert res == -1


def test_hyperplane_intersection_distance_intersect_rotate_z():
    initial_pt_basis = Point([0, 0, 0])
    basis = (Vector([1, 0, 0]), Vector([0, 1, 0]), Vector([0, 0, 1]))
    cs = CoordinateSystem(basis, initial_pt_basis)
    initial_pt_ray = Point([0, 1, 0])
    direction_ray = Vector([1, 0, 0])
    initial_pt_hyperelipsoid = Point([5, 1, 0])
    direction_hyperelipsoid = Vector([1, 0, 0])
    semiaxes = [1, 2, 3]
    ray = Ray(cs, initial_pt_ray, direction_ray)
    hyperplane = HyperElipsoid(cs, initial_pt_hyperelipsoid, semiaxes)
    for vec in hyperplane.cs.basis.basis:
        print(vec)
        print('--')
    hyperplane.rotate_3d(0, 0, math.radians(90))
    print('-------------------')
    for vec in hyperplane.cs.basis.basis:
        print(vec)
        print('--')
    res = hyperplane.intersection_distance(ray)
    print('res', res)
    assert res == 3


def test_hyperplane_intersection_distance_intersect_rotate_y():
    initial_pt_basis = Point([0, 0, 0])
    basis = (Vector([1, 0, 0]), Vector([0, 1, 0]), Vector([0, 0, 1]))
    cs = CoordinateSystem(basis, initial_pt_basis)
    initial_pt_ray = Point([0, 1, 0])
    direction_ray = Vector([1, 0, 0])
    initial_pt_hyperelipsoid = Point([5, 1, 0])
    direction_hyperelipsoid = Vector([1, 0, 0])
    semiaxes = [1, 2, 3]
    ray = Ray(cs, initial_pt_ray, direction_ray)
    hyperplane = HyperElipsoid(cs, initial_pt_hyperelipsoid, semiaxes)
    for vec in hyperplane.cs.basis.basis:
        print(vec)
        print('--')
    hyperplane.rotate_3d(0, math.radians(90), 0)
    print('-------------------')
    for vec in hyperplane.cs.basis.basis:
        print(vec)
        print('--')
    res = hyperplane.intersection_distance(ray)
    print('res', res)
    assert res == 2


def test_hyperplane_intersection_distance_intersect_rotate_x():
    initial_pt_basis = Point([0, 0, 0])
    basis = (Vector([1, 0, 0]), Vector([0, 1, 0]), Vector([0, 0, 1]))
    cs = CoordinateSystem(basis, initial_pt_basis)
    initial_pt_ray = Point([0, 1, 0])
    direction_ray = Vector([1, 0, 0])
    initial_pt_hyperelipsoid = Point([5, 1, 0])
    direction_hyperelipsoid = Vector([1, 0, 0])
    semiaxes = [1, 2, 3]
    ray = Ray(cs, initial_pt_ray, direction_ray)
    hyperplane = HyperElipsoid(cs, initial_pt_hyperelipsoid, semiaxes)
    for vec in hyperplane.cs.basis.basis:
        print(vec)
        print('--')
    hyperplane.rotate_3d(math.radians(90), 0, 0)
    print('-------------------')
    for vec in hyperplane.cs.basis.basis:
        print(vec)
        print('--')
    res = hyperplane.intersection_distance(ray)
    print('res', res)
    assert res == 4