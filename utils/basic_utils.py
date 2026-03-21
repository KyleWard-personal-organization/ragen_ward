import os
from dotenv import load_dotenv
import os.path as op
import functools
import time
from utils.logger import logger
import configs.constants as cs


def setup_env():
    """加载环境变量"""
    dotenv_path = op.join(cs.ROOT_DIR, '.env')
    load_dotenv(dotenv_path=dotenv_path)
    logger.info("Environment variables loaded")
    return


def calculate_time(func):
    """计算函数执行时间的装饰器"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = round(end_time - start_time, 3)
        logger.info(f"[calculate_time] The running time of ({func.__name__}) is {execution_time} seconds")
        return result
    return wrapper
