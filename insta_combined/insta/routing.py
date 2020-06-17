# from channels.routing import ProtocolTypeRouter

# application = ProtocolTypeRouter({
#     # (http->django views is added by default)
# })

from channels.auth import AuthMiddlewareStack
from channels.routing import ProtocolTypeRouter, URLRouter
import photo.routing

application = ProtocolTypeRouter({
    # (http->django views is added by default)
    'websocket': AuthMiddlewareStack(
        URLRouter(
            photo.routing.websocket_urlpatterns
        )
    ),
})