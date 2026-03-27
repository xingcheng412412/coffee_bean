import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from algorithm.particle_go import get_partical

# 测试函数
def test_get_partical():
    # 假设222.jpg在当前工作目录
    img_path = "222.jpg"
    try:
        with open(img_path, 'rb') as img_file:
            success, result = get_partical(img_file, detect_type=0, filename="222.jpg")
        if success:
            print("检测成功")
            print(result)
        else:
            print("检测失败:", result)
    except FileNotFoundError:
        print(f"文件 {img_path} 未找到")
    except Exception as e:
        print(f"测试出错: {e}")

if __name__ == "__main__":
    test_get_partical()