from lib.engine.engine import Identifier


# Заполнение множества идентификаторов
# ------------------------------------------------------------
def test_identifier_set_fill_1():
    identifier1 = Identifier()
    identifier2 = Identifier()
    assert len(identifier1.identifiers) == 2


def test_identifier_set_fill_2():
    identifier1 = Identifier()
    identifier2 = Identifier()
    assert len(identifier2.identifiers) == 4
