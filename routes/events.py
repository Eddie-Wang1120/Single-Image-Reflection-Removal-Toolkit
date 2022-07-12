from flask import session
from flask_socketio import emit, join_room, leave_room
from .define_socket import wjh_socketio
from model import testall

print('导入文件')


@wjh_socketio.on('connect', namespace='/uploadImage_ok')
def connect():
    print('连接成功')


@wjh_socketio.on('BDNstart', namespace='/uploadImage_ok')
def BDNstart(information):
    room_name = information.get('client_to_server')
    join_room(room_name)
    emit('status', {'server_to_client': ' BDNstart'}, room=room_name)


@wjh_socketio.on('BDNfinish', namespace='/uploadImage_ok')
def BDNfinish(information):
    import os
    room_name = information.get('client_to_server')
    testall.BDN()
    emit('status', {'server_to_client': ' BDNfinish'}, room=room_name)
    emit('onefinished', {'server_to_client': ' BDNfinish'}, room=room_name)
    testall.IBLNN()
    testall.IBCLN()
    testall.PRN()
    testall.RR()
    testall.resultResolute()
    os.system('rm -rf /home/wjh/Desktop/SIRR_Toolkit/static/downloads/*')
    f = open("/home/wjh/Desktop/SIRR_Toolkit/static/uploaded/image_name.txt")
    name = f.read()
    filename = name.split('.')[0]
    os.system('cp /home/wjh/Desktop/SIRR_Toolkit/model/output/BDN/B_1.jpg'
              + ' /home/wjh/Desktop/SIRR_Toolkit/static/downloads/BDN_res.jpg')

    os.system('cp ' + os.path.join('/home/wjh/Desktop/SIRR_Toolkit/model/output/IBCLN/test_final/images',
                                   filename + '_fake_Ts_03.jpg')
              + ' /home/wjh/Desktop/SIRR_Toolkit/static/downloads/IBCLN_res.jpg')
    os.system('cp ' + os.path.join('/home/wjh/Desktop/SIRR_Toolkit/model/output/PRN',
                                   'T_' + filename + '.jpg')
              + ' /home/wjh/Desktop/SIRR_Toolkit/static/downloads/PRN_res.jpg')
    os.system('cp ' + os.path.join('/home/wjh/Desktop/SIRR_Toolkit/model/output/RR/dataset',
                                   filename+'_pred_T.jpg')
              + ' /home/wjh/Desktop/SIRR_Toolkit/static/downloads/RR_res.jpg')
    os.system('cp ' + os.path.join('/home/wjh/Desktop/SIRR_Toolkit/model/output/IBLNN/IM2_1000_140/test_latest/images',
                                   filename + '_fake_Ts_03.jpg')
              + ' /home/wjh/Desktop/SIRR_Toolkit/static/downloads/IBLNN_res.jpg')


@wjh_socketio.on('BDNleft', namespace='/uploadImage_ok')
def BDNleft(information):
    room_name = information.get('client_to_server')
    leave_room(room_name)
    emit('status', {'server_to_client': ' BDNleft'}, room=room_name)


@wjh_socketio.on('Teststart', namespace='/uploadTest_ok')
def Teststart(information):
    room_name = information.get('client_to_server')
    join_room(room_name)
    emit('status', {'server_to_client': ' Teststart'}, room=room_name)


@wjh_socketio.on('Testfinish', namespace='/uploadTest_ok')
def Testfinish(information):
    import os
    room_name = information.get('client_to_server')
    testall.BDN()
    testall.IBLNN()
    emit('status', {'server_to_client': ' Testfinish'}, room=room_name)
    emit('Testfinished', {'server_to_client': ' Testfinish'}, room=room_name)
    testall.IBCLN()
    testall.PRN()
    testall.RR()
    testall.resultResolute()
    os.system('rm -rf /home/wjh/Desktop/SIRR_Toolkit/static/downloads/*')
    os.system('cp -r ./model/output ./static/downloads')



@wjh_socketio.on('Testleft', namespace='/uploadTest_ok')
def Testleft(information):
    room_name = information.get('client_to_server')
    leave_room(room_name)
    emit('status', {'server_to_client': ' Testleft'}, room=room_name)




@wjh_socketio.on('IBCLNstart', namespace='/uploadImage_ok')
def IBCLNstart(information):
    room_name = information.get('client_to_server')
    join_room(room_name)
    emit('status', {'server_to_client': ' IBCLNstart'}, room=room_name)


@wjh_socketio.on('IBCLNfinish', namespace='/uploadImage_ok')
def IBCLNfinish(information):
    room_name = information.get('client_to_server')
    testall.IBCLN()
    emit('status', {'server_to_client': ' IBCLNfinish'}, room=room_name)


@wjh_socketio.on('IBCLNleft', namespace='/uploadImage_ok')
def IBCLNleft(information):
    room_name = information.get('client_to_server')
    leave_room(room_name)
    emit('status', {'server_to_client': ' IBCLNleft'}, room=room_name)


@wjh_socketio.on('PRNstart', namespace='/uploadImage_ok')
def PRNstart(information):
    room_name = information.get('client_to_server')
    join_room(room_name)
    emit('status', {'server_to_client': ' PRNstart'}, room=room_name)


@wjh_socketio.on('PRNfinish', namespace='/uploadImage_ok')
def PRNfinish(information):
    room_name = information.get('client_to_server')
    testall.PRN()
    emit('status', {'server_to_client': ' PRNfinish'}, room=room_name)


@wjh_socketio.on('PRNleft', namespace='/uploadImage_ok')
def PRNleft(information):
    room_name = information.get('client_to_server')
    leave_room(room_name)
    emit('status', {'server_to_client': ' PRNleft'}, room=room_name)


@wjh_socketio.on('RRstart', namespace='/uploadImage_ok')
def RRstart(information):
    room_name = information.get('client_to_server')
    join_room(room_name)
    emit('status', {'server_to_client': ' RRstart'}, room=room_name)


@wjh_socketio.on('RRfinish', namespace='/uploadImage_ok')
def RRfinish(information):
    room_name = information.get('client_to_server')
    testall.RR()
    emit('status', {'server_to_client': ' RRfinish'}, room=room_name)


@wjh_socketio.on('RRleft', namespace='/uploadImage_ok')
def RRleft(information):
    room_name = information.get('client_to_server')
    leave_room(room_name)
    emit('status', {'server_to_client': ' RRleft'}, room=room_name)


@wjh_socketio.on('joined', namespace='/uploadImage_ok')
def joined(information):
    # 'joined'路由是传入一个room_name,给该websocket连接分配房间,返回一个'status'路由
    room_name = information.get('client_to_server')
    join_room(room_name)
    emit('status', {'server_to_client': ' enter the room'}, room=room_name)


@wjh_socketio.on('left', namespace='/uploadImage_ok')
def left(information):
    # 传入 room_name 输出 系统消息
    room_name = information.get('client_to_server')
    emit('status', {'server_to_client': ' has left the room'}, room=room_name)
    leave_room(room_name)


@wjh_socketio.on('text', namespace='/uploadImage_ok')
def text(information):
    # 传入 room_name, text 输出text，user_name
    room_name = information.get('client_to_server')
    text = information.get('text')
    emit('message', {
        'user_name': '',
        'text': text,
    }, room=room_name)
    emit('message', {
        'user_name': '',
        'text': text,
    }, room=room_name)
