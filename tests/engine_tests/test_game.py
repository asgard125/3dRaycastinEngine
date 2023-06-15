from lib.engine.engine import Entity, EntitiesList, Game
from lib.math.base_classes import Point, CoordinateSystem


# Инициализация Game
# ------------------------------------------------------------
def test_game_init():
    initial_pt = Point([0, 0, 0])
    basis = ([1, 0, 0], [0, 1, 0], [0, 0, 1])
    cs = CoordinateSystem(basis, initial_pt)
    initial_pt = Point([0, 0, 0])
    basis = ([1, 0, 0], [0, 1, 0], [0, 0, 1])
    cs = CoordinateSystem(basis, initial_pt)
    entity1 = Entity(cs)
    entity2 = Entity(cs)
    entity3 = Entity(cs)
    entities_list = EntitiesList(entity1, entity2, entity3)
    assert Game(cs, entities_list)


# Создание класов с заданной системой координат
# ------------------------------------------------------------
def test_game_init_entity():
    initial_pt = Point([0, 0, 0])
    basis = ([1, 0, 0], [0, 1, 0], [0, 0, 1])
    cs = CoordinateSystem(basis, initial_pt)
    initial_pt = Point([0, 0, 0])
    basis = ([1, 0, 0], [0, 1, 0], [0, 0, 1])
    cs = CoordinateSystem(basis, initial_pt)
    entity1 = Entity(cs)
    entity2 = Entity(cs)
    entity3 = Entity(cs)
    entities_list = EntitiesList(entity1, entity2, entity3)
    game = Game(cs, entities_list)
    game_entity = game.get_entity_class()
    assert game_entity.cs == cs