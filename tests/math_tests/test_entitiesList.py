from lib.engine.engine import Entity, EntitiesList
from lib.math.base_classes import Point, CoordinateSystem
import pytest
from lib.exceptions.engine_exceptions import *


# Инициализация EntitiesList
# ------------------------------------------------------------
def test_entitieslist_init():
    initial_pt = Point([0, 0, 0])
    basis = ([1, 0, 0], [0, 1, 0], [0, 0, 1])
    cs = CoordinateSystem(basis, initial_pt)
    entity1 = Entity(cs)
    entity2 = Entity(cs)
    entity3 = Entity(cs)
    assert EntitiesList(entity1, entity2, entity3)


# Добавление Entity
# ------------------------------------------------------------
def test_entitieslist_append():
    initial_pt = Point([0, 0, 0])
    basis = ([1, 0, 0], [0, 1, 0], [0, 0, 1])
    cs = CoordinateSystem(basis, initial_pt)
    entity1 = Entity(cs)
    entity2 = Entity(cs)
    entity3 = Entity(cs)
    entities_list = EntitiesList(entity1, entity2)
    entities_list.append(entity3)
    assert entities_list.entities[-1] == entity3


def test_entitieslist_append_already_in_list():
    initial_pt = Point([0, 0, 0])
    basis = ([1, 0, 0], [0, 1, 0], [0, 0, 1])
    cs = CoordinateSystem(basis, initial_pt)
    entity1 = Entity(cs)
    entity2 = Entity(cs)
    entity3 = Entity(cs)
    entities_list = EntitiesList(entity1, entity2)
    entities_list.append(entity3)
    with pytest.raises(EntitiesListEntityInListException):
        entities_list.append(entity3)


# Удаление Entity
# ------------------------------------------------------------
def test_entitieslist_remove():
    initial_pt = Point([0, 0, 0])
    basis = ([1, 0, 0], [0, 1, 0], [0, 0, 1])
    cs = CoordinateSystem(basis, initial_pt)
    entity1 = Entity(cs)
    entity2 = Entity(cs)
    entity3 = Entity(cs)
    entities_list = EntitiesList(entity1, entity2)
    entities_list.append(entity3)
    entities_list.remove(entity3.identifier)
    assert entities_list.entities[-1] != entity3


def test_entitieslist_append_not_in_list():
    initial_pt = Point([0, 0, 0])
    basis = ([1, 0, 0], [0, 1, 0], [0, 0, 1])
    cs = CoordinateSystem(basis, initial_pt)
    entity1 = Entity(cs)
    entity2 = Entity(cs)
    entity3 = Entity(cs)
    entities_list = EntitiesList(entity1, entity2)
    with pytest.raises(EntitiesListEntityNotInListException):
        entities_list.remove(entity3.identifier)


# Получение Entity
# ------------------------------------------------------------
def test_entitieslist_get():
    initial_pt = Point([0, 0, 0])
    basis = ([1, 0, 0], [0, 1, 0], [0, 0, 1])
    cs = CoordinateSystem(basis, initial_pt)
    entity1 = Entity(cs)
    entity2 = Entity(cs)
    entities_list = EntitiesList(entity1, entity2)
    assert entities_list.get(entity2.identifier) == entity2


def test_entitieslist_get_not_in_list():
    initial_pt = Point([0, 0, 0])
    basis = ([1, 0, 0], [0, 1, 0], [0, 0, 1])
    cs = CoordinateSystem(basis, initial_pt)
    entity1 = Entity(cs)
    entity2 = Entity(cs)
    entity3 = Entity(cs)
    entities_list = EntitiesList(entity1, entity2)
    assert entities_list.get(entity3.identifier) is None


# Выполнение функций на списке
# ------------------------------------------------------------
def set_property(entity):
    entity.power = 3


def test_entitieslist_exec():
    initial_pt = Point([0, 0, 0])
    basis = ([1, 0, 0], [0, 1, 0], [0, 0, 1])
    cs = CoordinateSystem(basis, initial_pt)
    entity1 = Entity(cs)
    entity2 = Entity(cs)
    entity3 = Entity(cs)
    entities_list = EntitiesList(entity1, entity2, entity3)
    entities_list.exec(set_property)
    assert all([e.power == 3 for e in entities_list.entities])