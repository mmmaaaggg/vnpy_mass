"""
@author  : MG
@Time    : 2020/9/2 16:58
@File    : config.py
@contact : mmmaaaggg@163.com
@desc    : 用于配置文件
"""
import logging
from logging.config import dictConfig

# log settings
logging_config = dict(
    version=1,
    formatters={
        'simple': {
            'format': '%(asctime)s %(name)s|%(module)s.%(funcName)s:%(lineno)d %(levelname)s %(message)s'}
    },
    handlers={
        'file_handler':
            {
                'class': 'logging.handlers.RotatingFileHandler',
                'filename': 'logger.log',
                'maxBytes': 1024 * 1024 * 10,
                'backupCount': 5,
                'level': 'INFO',
                'formatter': 'simple',
                'encoding': 'utf8'
            },
        'console_handler':
            {
                'class': 'logging.StreamHandler',
                'level': 'INFO',
                'formatter': 'simple'
            }
    },

    root={
        'handlers': ['console_handler', 'file_handler'],
        'level': logging.INFO,
    }
)
dictConfig(logging_config)
logging.info("加载配置文件")

if __name__ == "__main__":
    pass
