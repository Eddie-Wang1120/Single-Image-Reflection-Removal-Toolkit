import tarfile

from flask import (
    redirect,
    request,
    session,
    render_template,
    url_for,
    Blueprint,
    make_response, jsonify, send_from_directory,
)
from werkzeug.utils import secure_filename

main = Blueprint('main', __name__)

@main.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@main.route('/return_to_index')
def return_to_index():
    return render_template('index.html')

@main.route('/oneresult')
def oneresult():
    return render_template('oneresult.html')

@main.route('/zipresult')
def zipresult():
    return render_template('zipresult.html')

@main.route('/download_one')
def download_one():
    import os
    os.chdir('static/downloads')
    os.system('zip results.zip *')
    os.chdir('..')
    os.chdir('..')
    return send_from_directory(r"/home/wjh/Desktop/SIRR_Toolkit/static/downloads", "results.zip", as_attachment=True)

@main.route('/download_zip')
def download_zip():
    import os
    os.chdir('static/downloads')
    os.system('zip -r results.zip *')
    os.chdir('..')
    os.chdir('..')
    return send_from_directory(r"/home/wjh/Desktop/SIRR_Toolkit/static/downloads", "results.zip", as_attachment=True)

@main.route('/uploadImage',methods=['GET'])
def nextImage():
    res = make_response(render_template('uploadImage.html'))
    return res

@main.route('/uploadTest',methods=['GET'])
def nextTest():
    res = make_response(render_template('uploadTest.html'))
    return res

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@main.route('/upload', methods=['POST', 'GET'])  # 添加路由
def upload():
    import os
    import cv2
    import time
    if request.method == 'POST':
        f = request.files['file']
        name = f.filename
        os.system('rm -rf /home/wjh/Desktop/SIRR_Toolkit/static/uploaded/*')
        the = open("/home/wjh/Desktop/SIRR_Toolkit/static/uploaded/image_name.txt", "w")
        the.write(name)
        the.close()

        user_input = request.form.get("name")
        os.system('rm -rf /home/wjh/Desktop/SIRR_Toolkit/model/dataset/R/*')
        os.system('rm -rf /home/wjh/Desktop/SIRR_Toolkit/model/dataset/I/*')
        os.system('rm -rf /home/wjh/Desktop/SIRR_Toolkit/model/dataset/B/*')

        upload_path = os.path.join('/home/wjh/Desktop/SIRR_Toolkit/model/dataset/B', name)  # 注意：没有的文件夹一定要先创建，不然会提示没有该路径
        f.save(upload_path)

        os.system('cp ' + os.path.join('/home/wjh/Desktop/SIRR_Toolkit/model/dataset/B', name) +
                  ' /home/wjh/Desktop/SIRR_Toolkit/model/dataset/I')
        os.system('cp ' + os.path.join('/home/wjh/Desktop/SIRR_Toolkit/model/dataset/B', name) +
                  ' /home/wjh/Desktop/SIRR_Toolkit/model/dataset/R')
        # 使用Opencv转换一下图片格式和名称
        img = cv2.imread(upload_path)
        cv2.imwrite(os.path.join('/home/wjh/Desktop/SIRR_Toolkit/static/uploaded', 'test.jpg'), img)


        return render_template('uploadImage_ok.html', userinput=user_input, val1=time.time())

    return render_template('uploadImage.html')


@main.route('/uploadZip', methods=['POST', 'GET'])  # 添加路由
def uploadZip():
    import os
    import time
    if request.method == 'POST':
        f = request.files['file']
        user_input = request.form.get("name")
        os.system('rm -rf /home/wjh/Desktop/SIRR_Toolkit/static/uploaded/*')
        os.system('rm -rf /home/wjh/Desktop/SIRR_Toolkit/model/dataset/*')
        upload_path = os.path.join('/home/wjh/Desktop/SIRR_Toolkit/model/dataset', f.filename)
        f.save(upload_path)

        os.chdir('model/dataset')
        os.system('unzip dataset.zip')
        os.chdir('..')
        os.chdir('..')

        return render_template('uploadTest_ok.html', userinput=user_input, val1=time.time())

    return render_template('uploadTest.html')

