import mxnet as mx

from pathlib import Path

from kigo.dl.betago import BetaGoAgent
from kigo.http.server import get_web_app

checkpoint_p = Path('./checkpoints/').resolve().joinpath('betago.params')
ctx = mx.gpu()
agent = BetaGoAgent.create(checkpoint_p, ctx)
web_app = get_web_app({'betago': agent})
# NOTE: using multuithreaded FLASK results in a CUDNN_STATUS_MAPPING_ERROR in MXNet
# see: https://github.com/apache/incubator-mxnet/issues/3946
web_app.run(threaded=False, processes=1)
