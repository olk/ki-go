class Response:
    def __init__(self, status, body):
        self.success = status
        self.body = body


# successful GTP response with response body
def success(body=''):
    return Response(status=True, body=body)


# error GTP response
def error(body=''):
    return Response(status=False, body=body)


# convert a Python boolean into GTP
def bool_response(boolean):
    return success('true') if boolean is True else success('false')


# serialize a GTP response as a string
def serialize(gtp_command, gtp_response):
    return '{}{} {}\n\n'.format(
        '=' if gtp_response.success else '?',
        '' if gtp_command.sequence is None else str(gtp_command.sequence),
        gtp_response.body)
