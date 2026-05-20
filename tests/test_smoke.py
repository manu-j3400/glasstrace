"""

Smoke tests - package import verification and exposes the expected surface

"""

import glasstrace


def test_import():
    """ Checks for package import errors and empty packages  """
    assert glasstrace is not None

def test_version():
    """ package version validation  """
    assert isinstance(glasstrace.__version__, str)
    assert len(glasstrace.__version__) > 0
