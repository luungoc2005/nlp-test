
from flask import jsonify

def get_json(request):
    content = request.get_json()
    if content is None:
        try:
            import json
            content = json.loads(content, 'utf8')
        except:
            pass
    return content

def allowed_file(filename, allowed_exts=['json']):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_exts

def jsonerror(*args, **kwargs):
    response = jsonify(*args, **kwargs)
    response.status_code = 400
    return response
