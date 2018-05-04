import unittest
import json
from os import environ
from os import urandom

# Defer any configuration to the tests setUp()
environ['DAEMON_DEFER_CONFIG'] = "True"

import daemon


# Set up TESTING and DEBUG env vars to be picked up by Flask
daemon.app.config['DEBUG'] = True
daemon.app.config['TESTING'] = True
# Set a random secret key for testing
daemon.app.config['SECRET_KEY'] = str(urandom(32))

# Duplicate app config settings into the bp, like the register would
daemon.blueprint.BLUEPRINT.config['DEBUG'] = True
daemon.blueprint.BLUEPRINT.config['TESTING'] = True
daemon.blueprint.BLUEPRINT.config['SECRET_KEY'] = \
    daemon.app.config['SECRET_KEY']


class Tests(unittest.TestCase):
    def setUp(self):
        # Perform any setup that should occur
        # before every test
        self.app = daemon.app.test_client()


    def tearDown(self):
        # Perform any tear down that should
        # occur after every test
        del self.app

    def testPass(self):
        self.assertEqual(True, True)

    def testVersionAvailable(self):
        x = getattr(daemon, "__version__", None)
        self.assertTrue(x is not None)

    def testVersion(self):
        version_response = self.app.get("/version")
        self.assertEqual(version_response.status_code, 200)
        version_json = json.loads(version_response.data.decode())
        api_reported_version = version_json['version']
        self.assertEqual(
            daemon.blueprint.__version__,
            api_reported_version
        )


if __name__ == "__main__":
    unittest.main()
