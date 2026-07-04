import configparser
import importlib.util
from pathlib import Path


def test_pytest_timeout_options_require_timeout_plugin():
    parser = configparser.ConfigParser()
    parser.read(Path(__file__).resolve().parents[1] / "pytest.ini")
    pytest_section = parser["pytest"] if parser.has_section("pytest") else {}

    declares_timeout_options = any(key in pytest_section for key in ("timeout", "timeout_method"))
    has_pytest_timeout = importlib.util.find_spec("pytest_timeout") is not None

    assert has_pytest_timeout or not declares_timeout_options
