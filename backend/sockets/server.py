import socketio

# Create a Socket.IO server with CORS enabled
sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')

@sio.event
async def connect(sid, environ):
    print(f"Client connected: {sid}")

@sio.event
async def disconnect(sid):
    print(f"Client disconnected: {sid}")

@sio.event
async def join_factory(sid, factory_id):
    sio.enter_room(sid, factory_id)
    print(f"Client {sid} joined factory room: {factory_id}")

@sio.event
async def join_line(sid, line_id):
    sio.enter_room(sid, line_id)
    print(f"Client {sid} joined line room: {line_id}")
