from ecosys.entities import Entity

def test_distance():
    ent1 = Entity((0, 0))
    ent2 = Entity((1, 0))
    assert ent1.distance(ent2) == 1