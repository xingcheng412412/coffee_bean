import argparse
import hashlib
import httpx
import os
import logging

def calculate_sha1(file_data):
    """计算文件的SHA1值"""
    return hashlib.sha1(file_data).hexdigest()

def get_content_type(file_path):
    """根据文件后缀获取Content-Type"""
    # 文件后缀到Content-Type的映射
    extension_to_content_type = {
        # 图片
        'gif': 'image/gif',
        'jpeg': 'image/jpeg',
        'jpg': 'image/jpg',
        'png': 'image/png',
        'bmp': 'image/bmp',
        'webp': 'image/webp',
        'heic': 'image/heic',
        'heif': 'image/heif',

        # 视频
        'mp4': 'video/mp4',
        'mkv': 'video/x-matroska',
        'avi': 'video/x-msvideo',
        '3gp': 'video/3gpp',
        'flv': 'video/x-flv',
        'mpg': 'video/mpeg',
        'mov': 'video/quicktime',
        'wmv': 'video/x-ms-wmv',
        'm3u8': 'application/vnd.apple.mpegurl',
        
        # 音频
        'amr': 'audio/amr',
        'mp3': 'audio/mpeg',
        'wav': 'audio/wav',
        'm4a': 'audio/x-m4a',
        'wmv': 'audio/wmv',
        'flv': 'audio/flv',
        'flac': 'audio/flac',
        'aac': 'audio/aac',
        'm4a': 'audio/x-m4a',
        'm4b': 'audio/x-m4b',
        'm4p': 'audio/x-m4p',
        'm4v': 'video/x-m4v',
        
        # 文档类型
        'pdf': 'application/pdf',
        'doc': 'application/msword',
        'ppt': 'application/vnd.ms-powerpoint',
        'xls': 'application/vnd.ms-excel',
        'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
        'xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        
        # 文本文件
        'txt': 'text/plain',
        'csv': 'text/csv',
        'xml': 'text/xml',
        'json': 'application/json',
        'bin': 'application/octet-stream',
        
        # 压缩包
        'zip': 'application/zip',
        '7z': 'application/x-7z-compressed',
        'tar': 'application/x-tar',
        'gz': 'application/x-gzip',
        'bz2': 'application/x-bzip2',
        'rar': 'application/x-rar-compressed',
        
        # 其他
        'apk': 'application/vnd.android.package-archive',
        'ipa': 'application/iphone-package-archive'
    }
    
    # 获取文件后缀（转换为小写）
    ext = os.path.splitext(file_path)[1].lower().lstrip('.')
    
    # 返回对应的Content-Type，如果没有找到则返回默认值
    return extension_to_content_type.get(ext, 'application/octet-stream')

def upload_file(file_path, app_id, task_id, token, server="http://127.0.0.1:2090", max_length=0):
    """同步上传文件到服务器"""
    # 读取文件内容
    with open(file_path, 'rb') as f:
        file_data = f.read()
    # 计算文件的SHA1值
    file_hash = calculate_sha1(file_data)
    # 获取正确的Content-Type
    content_type = get_content_type(file_path)
    # 设置请求头
    headers = {
        'X-AppID': app_id,
        'X-Hash': file_hash,
        'X-TaskId': task_id,
        'X-Token': token,
        'Content-Type': content_type
    }
    if max_length > 0:
        headers['X-resize'] = f'max={max_length}'
    try:
        # 使用同步客户端发送POST请求
        with httpx.Client() as client:
            response = client.post(
                server + '/upload/particle/simple',
                headers=headers,
                data=file_data
            )
            logging.info(f"上传文件: {file_path}, 状态码: {response.status_code}, 响应: {response.text}")
            # 检查响应状态
            if response.status_code == 200:
                result = response.json()
                success = result.get("code", 0) == 200
                errmsg = result.get("msg", "")
                if success:
                    return True, result.get("data", {})
                else:
                    return False, errmsg
            else:
                return False, f"上传失败！状态码: {response.status_code}"
    except Exception as e:
        logging.error(f"上传'{file_path}'过程中发生错误: {str(e)}")
        return False, f"上传过程中发生错误: {str(e)}"

def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='文件上传客户端')
    parser.add_argument('--app-id', required=False, default="particle", help='应用ID')
    parser.add_argument('--server', required=False, default='http://127.0.0.1:2090', help='服务器URL，例如: http://example.com')
    parser.add_argument('--file', required=True, help='要上传的文件路径')
    parser.add_argument('--task-id', required=True, help='任务ID')
    parser.add_argument('--token', required=True, help='认证令牌')
    parser.add_argument('--max-length', type=int, required=False, default=0, help='最大长度')

    # 解析命令行参数
    args = parser.parse_args()

    # 检查文件是否存在
    if not os.path.exists(args.file):
        print(f"错误：文件 '{args.file}' 不存在")
        return

    # 同步上传文件
    ok, result = upload_file(args.file, args.app_id, args.task_id, args.token,
                             server=args.server, max_length=args.max_length)
    if ok:
        print(f"文件上传成功！")
        print(f"文件URL: {result.get('url')}")
        print(f"文件RID: {result.get('rid')}")
    else:
        print(f"文件上传失败！错误信息: {result}")

"""
管理后台token: T1vxb8f_f8kj-X9BlTteCquWx-U7z0VCTpA18jn1I3HDr6qwOkxn
"""
if __name__ == '__main__':
    main()
