import logging
import numpy as np
from algorithm.coffee_powder_Info import *
from algorithm.coffee_bean_Info import *
from pydantic import BaseModel
import traceback

class MParticleResult(BaseModel):
    ave_particle_size: float | None = None           # 平均粒径（所有粒径的平均值，单位：μm）
    mid_particle_size: float | None = None           # D50粒径（50%百分位数，单位：μm）
    high_particle_size: float | None = None          # D97粒径（97%百分位数，单位：μm）
    ave_area: float | None = None                    # 平均面积（平均粒径的平方乘以π，单位：μm²）
    most_frequent_particle_size: float | None = None # 最频粒径（分布中频率最高的粒径值，单位：μm）
    pass_rate: float | None = None                   # 杯测通过率（200-850nm区间所占百分比，单位：%）
    fine_particles: float | None = None              # 细粉比例（0-200μm区间所占百分比，单位：%）
    medium_particles: float | None = None            # 中等粒径比例（200-1000μm区间所占百分比，单位：%）
    large_particles: float | None = None             # 大粒径比例（1000μm以上区间所占百分比，单位：%）
    result: dict | None = None                       # 其他分析结果（预留字段，字典类型）
    particle_x: list | None = None                   # 粒径分布X轴数据（粒径区间列表）（咖啡豆目数）
    particle_y: list | None = None                   # 粒径分布Y轴数据（对应区间的数量或频率）
    particle_y_accumulate: list | None = None        # 粒径分布累计Y轴数据（累计百分比）
    share_report: str | None = None                  # 结果报告分享链接
    bean_detect_result: dict | None = None           # 咖啡豆检测结果（如瑕疵检测等，字典类型）


def get_partical(
    img_file,      # 图片对象
    detect_type=0, # 检测类型：可选参数，默认值为0,0-咖啡粉粒径检测，2-咖啡豆粒径检测
    filename=None # 文件名：可选参数，默认值为None
):

    try:
        if(detect_type==0):
            x_data, y_data, y_data_accumulate, digit, res_url= get_coffee_powder_Info(img_file)
            logging.info(f"图片 filename:{filename} detect_type:{detect_type} 检测成功")
            """ 平均粒径 所有粒径的一个平均数 """
            ave_particle_size = np.round(digit[0], 0)
            """ 面积就是 平均粒径^2*pi """
            ave_area = np.round(digit[1], 0)
            """ 最频粒径 分布中频率最高的那个对应的粒径值 """
            most_frequent_particle_size = np.round(digit[2], 0)
            """ D50  D97  是分别是 50%百分位数  和 97 百分位数  """
            mid_particle_size = np.round(digit[3], 0)
            high_particle_size = np.round(digit[4], 0)
            """ 杯测通过率 200-850um 所占整体的百分比    """
            passRate = np.round(digit[5], 2)
            """ 细粉比例 0-200um 所占整体的百分比    """
            fine_particles = np.round(digit[6], 2)
            """ 杯测通过率 200-1000um 所占整体的百分比    """
            medium_particles = np.round(digit[7], 2)
            """ 杯测通过率 1000um以上 所占整体的百分比    """
            large_particles = np.round(digit[8], 2)
            res = MParticleResult(
                             ave_particle_size=ave_particle_size, mid_particle_size=mid_particle_size,
                             high_particle_size=high_particle_size, ave_area=ave_area,
                             most_frequent_particle_size=most_frequent_particle_size, pass_rate=passRate,fine_particles=fine_particles,medium_particles=medium_particles,large_particles=large_particles,
                             particle_x=x_data, particle_y=y_data, particle_y_accumulate=y_data_accumulate,
                             share_report=res_url)

            return True, res
        
        if(detect_type==1):
            ave_particle_size, res_url, bean_detect_result, x_data, y_data, y_accumulate, _, _, _, _ = get_coffee_bean_Info(img_file)
            res = MParticleResult(ave_particle_size=ave_particle_size, bean_detect_result=bean_detect_result,
                                  particle_x=x_data, particle_y=y_data, particle_y_accumulate=y_accumulate)
            return True, res

        if(detect_type==2):
            ave_particle_size, res_url, bean_detect_result, x_data, y_data, y_accumulate, _, _, _, _ = get_coffee_bean_Info(img_file)
            res = MParticleResult(ave_particle_size=ave_particle_size, bean_detect_result=bean_detect_result,
                                  particle_x=x_data, particle_y=y_data, particle_y_accumulate=y_accumulate)
            return True, res
            
    except Exception as e:
        logging.error(f"检测失败({filename}): {traceback.format_exc()}")
        return False, "检测失败"