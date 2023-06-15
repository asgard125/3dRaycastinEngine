from lib.engine.engine import Entity
from lib.math.base_classes import Point, CoordinateSystem
import pytest
from lib.exceptions.engine_exceptions import *


# Инициализация Entity
# ------------------------------------------------------------
def test_entity_init():
    initial_pt = Point([0, 0, 0])
    basis = ([1, 0, 0], [0, 1, 0], [0, 0, 1])
    cs = CoordinateSystem(basis, initial_pt)
    assert Entity(cs)


# получение, удаление, добавление property через функции
# ------------------------------------------------------------
def test_entity_set_get_property_by_func():
    initial_pt = Point([0, 0, 0])
    basis = ([1, 0, 0], [0, 1, 0], [0, 0, 1])
    cs = CoordinateSystem(basis, initial_pt)
    entity = Entity(cs)
    prop = 'power'
    value = 1
    entity.set_property(prop, value)
    assert entity.get_property(prop) == value


def test_entity_set_reserved_property_by_func_1():
    initial_pt = Point([0, 0, 0])
    basis = ([1, 0, 0], [0, 1, 0], [0, 0, 1])
    cs = CoordinateSystem(basis, initial_pt)
    entity = Entity(cs)
    prop = 'cs'
    value = 1
    with pytest.raises(EntityReservedPropertyException):
        entity.set_property(prop, value)


def test_entity_set_reserved_property_by_func_2():
    initial_pt = Point([0, 0, 0])
    basis = ([1, 0, 0], [0, 1, 0], [0, 0, 1])
    cs = CoordinateSystem(basis, initial_pt)
    entity = Entity(cs)
    prop = 'properties'
    value = 1
    with pytest.raises(EntityReservedPropertyException):
        entity.set_property(prop, value)


def test_entity_set_reserved_property_by_func_3():
    initial_pt = Point([0, 0, 0])
    basis = ([1, 0, 0], [0, 1, 0], [0, 0, 1])
    cs = CoordinateSystem(basis, initial_pt)
    entity = Entity(cs)
    prop = 'identifier'
    value = 1
    with pytest.raises(EntityReservedPropertyException):
        entity.set_property(prop, value)


def test_entity_del_property_by_func():
    initial_pt = Point([0, 0, 0])
    basis = ([1, 0, 0], [0, 1, 0], [0, 0, 1])
    cs = CoordinateSystem(basis, initial_pt)
    entity = Entity(cs)
    prop = 'power'
    value = 1
    entity.set_property(prop, value)
    entity.remove_property(prop)
    with pytest.raises(EntityNonExistingPropertyException):
        entity.get_property(prop)


def test_entity_del_non_existing_property_by_func():
    initial_pt = Point([0, 0, 0])
    basis = ([1, 0, 0], [0, 1, 0], [0, 0, 1])
    cs = CoordinateSystem(basis, initial_pt)
    entity = Entity(cs)
    prop = 'power'
    with pytest.raises(EntityNonExistingPropertyException):
        entity.remove_property(prop)


def test_entity_del_reserved_property_by_func_1():
    initial_pt = Point([0, 0, 0])
    basis = ([1, 0, 0], [0, 1, 0], [0, 0, 1])
    cs = CoordinateSystem(basis, initial_pt)
    entity = Entity(cs)
    prop = 'cs'
    with pytest.raises(EntityReservedPropertyException):
        entity.remove_property(prop)


def test_entity_del_reserved_property_by_func_2():
    initial_pt = Point([0, 0, 0])
    basis = ([1, 0, 0], [0, 1, 0], [0, 0, 1])
    cs = CoordinateSystem(basis, initial_pt)
    entity = Entity(cs)
    prop = 'properties'
    with pytest.raises(EntityReservedPropertyException):
        entity.remove_property(prop)


def test_entity_del_reserved_property_by_func_3():
    initial_pt = Point([0, 0, 0])
    basis = ([1, 0, 0], [0, 1, 0], [0, 0, 1])
    cs = CoordinateSystem(basis, initial_pt)
    entity = Entity(cs)
    prop = 'identifier'
    with pytest.raises(EntityReservedPropertyException):
        entity.remove_property(prop)


# получение, удаление, добавление property через обращение через скобки
# ------------------------------------------------------------
def test_entity_set_get_property_by_ind():
    initial_pt = Point([0, 0, 0])
    basis = ([1, 0, 0], [0, 1, 0], [0, 0, 1])
    cs = CoordinateSystem(basis, initial_pt)
    entity = Entity(cs)
    prop = 'power'
    value = 1
    entity[prop] = value
    assert entity[prop] == value


def test_entity_set_reserved_property_by_ind_1():
    initial_pt = Point([0, 0, 0])
    basis = ([1, 0, 0], [0, 1, 0], [0, 0, 1])
    cs = CoordinateSystem(basis, initial_pt)
    entity = Entity(cs)
    prop = 'cs'
    value = 1
    with pytest.raises(EntityReservedPropertyException):
        entity[prop] = value


def test_entity_set_reserved_property_by_ind_2():
    initial_pt = Point([0, 0, 0])
    basis = ([1, 0, 0], [0, 1, 0], [0, 0, 1])
    cs = CoordinateSystem(basis, initial_pt)
    entity = Entity(cs)
    prop = 'properties'
    value = 1
    with pytest.raises(EntityReservedPropertyException):
        entity[prop] = value


def test_entity_set_reserved_property_by_ind_3():
    initial_pt = Point([0, 0, 0])
    basis = ([1, 0, 0], [0, 1, 0], [0, 0, 1])
    cs = CoordinateSystem(basis, initial_pt)
    entity = Entity(cs)
    prop = 'identifier'
    value = 1
    with pytest.raises(EntityReservedPropertyException):
        entity[prop] = value


def test_entity_del_property_by_ind():
    initial_pt = Point([0, 0, 0])
    basis = ([1, 0, 0], [0, 1, 0], [0, 0, 1])
    cs = CoordinateSystem(basis, initial_pt)
    entity = Entity(cs)
    prop = 'power'
    value = 1
    entity[prop] = value
    del entity[prop]
    with pytest.raises(EntityNonExistingPropertyException):
        res = entity[prop]


def test_entity_del_non_existing_property_by_ind():
    initial_pt = Point([0, 0, 0])
    basis = ([1, 0, 0], [0, 1, 0], [0, 0, 1])
    cs = CoordinateSystem(basis, initial_pt)
    entity = Entity(cs)
    prop = 'power'
    with pytest.raises(EntityNonExistingPropertyException):
        del entity[prop]


def test_entity_del_reserved_property_by_ind_1():
    initial_pt = Point([0, 0, 0])
    basis = ([1, 0, 0], [0, 1, 0], [0, 0, 1])
    cs = CoordinateSystem(basis, initial_pt)
    entity = Entity(cs)
    prop = 'cs'
    with pytest.raises(EntityReservedPropertyException):
        del entity[prop]


def test_entity_del_reserved_property_by_ind_2():
    initial_pt = Point([0, 0, 0])
    basis = ([1, 0, 0], [0, 1, 0], [0, 0, 1])
    cs = CoordinateSystem(basis, initial_pt)
    entity = Entity(cs)
    prop = 'properties'
    with pytest.raises(EntityReservedPropertyException):
        del entity[prop]


def test_entity_del_reserved_property_by_ind_3():
    initial_pt = Point([0, 0, 0])
    basis = ([1, 0, 0], [0, 1, 0], [0, 0, 1])
    cs = CoordinateSystem(basis, initial_pt)
    entity = Entity(cs)
    prop = 'identifier'
    with pytest.raises(EntityReservedPropertyException):
        del entity[prop]


# получение, удаление, добавление property через обращение как к атрибуту
# ------------------------------------------------------------
def test_entity_set_get_property_by_atr():
    initial_pt = Point([0, 0, 0])
    basis = ([1, 0, 0], [0, 1, 0], [0, 0, 1])
    cs = CoordinateSystem(basis, initial_pt)
    entity = Entity(cs)
    prop = 'power'
    value = 1
    entity.power = value
    assert entity.power == value


def test_entity_del_property_by_atr():
    initial_pt = Point([0, 0, 0])
    basis = ([1, 0, 0], [0, 1, 0], [0, 0, 1])
    cs = CoordinateSystem(basis, initial_pt)
    entity = Entity(cs)
    prop = 'power'
    value = 1
    entity.power = value
    del entity.power
    with pytest.raises(AttributeError):
        res = entity.power


def test_entity_del_non_existing_property_by_atr():
    initial_pt = Point([0, 0, 0])
    basis = ([1, 0, 0], [0, 1, 0], [0, 0, 1])
    cs = CoordinateSystem(basis, initial_pt)
    entity = Entity(cs)
    prop = 'power'
    with pytest.raises(AttributeError):
        del entity.power


