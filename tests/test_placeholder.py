from fiap_tech_challenge_4.utils.calculator import add


def test_placeholder():
    """A placeholder test to ensure that pytest works."""
    print("This is a test!")
    assert True


def test_add():
    """Test if the import is working."""
    assert add(2, 4) == 6
