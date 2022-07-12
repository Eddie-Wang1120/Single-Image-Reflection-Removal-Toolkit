#!/bin/env python
from flask import Flask
from routes.define_socket import wjh_socketio


def create_app(debug=True):
    app = Flask(__name__)
    app.debug = debug

    from routes.main_route import main as main_blueprint
    app.register_blueprint(main_blueprint)
    wjh_socketio.init_app(app)
    from routes import events
    return app


if __name__ == '__main__':
    app = create_app()

    wjh_socketio.run(app)











