from ecosys.entities import Entity

def test_distance():
    ent1 = Entity((0, 0))
    ent2 = Entity((1, 1))
    ent3 = Entity((1, 2))
    assert ent1.distance(ent1) == 0
    assert ent1.distance(ent2) == 2
    assert ent1.distance(ent3) == 3
    assert ent2.distance(ent3) == 1
