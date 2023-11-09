def pytest_addoption(parser):
    parser.addoption("--model", action="store", default=None, help="Model name to test")
