class Error(Exception):
    err_name = "Error"
    status_code = 500
    message = ""

    def __init__(self, message=None):
        if message is not None:
            self.message = message

    def to_dict(self):
        return {"message": self.message,
                "error_name": self.err_name}


# Exceptions which inherit from Error go here.
# Note that they will only be handled correctly if
# the included app errorhandler is used, or if whatever
# application mounts the blueprint implements a similar
# error handler.
