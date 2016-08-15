# coding:utf-8
from typing import Dict
from typing import List
from typing import Tuple
from typing import TypeVar

__auth__ = 'di_shen_sh@gmail.com'

T = TypeVar('T')


class StringException(Exception):
    def __init__(self, a_str:str, *args, **kwargs):
        # 注意不能使用a_str.format(args, kwargs)
        # 否则无法正常format
        self.info = a_str.format(*args, **kwargs)
    def __str__(self):
        return self.info




