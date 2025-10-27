import logging
import myutils


def configure_logger(log_file_path):
    LOG_LEVEL = "INFO"
    LOG_LEVEL_ATTRI = getattr(logging, LOG_LEVEL.upper(), None)
    if not isinstance(LOG_LEVEL_ATTRI, int):
        raise ValueError("Invalid log level: %s" % LOG_LEVEL)
    LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"
    if log_file_path:
        logging.basicConfig(
            filename=log_file_path, level=LOG_LEVEL_ATTRI, format=LOG_FORMAT
        )
    else:
        logging.basicConfig(level=LOG_LEVEL_ATTRI, format=LOG_FORMAT)


def mosaic_fraction_logger(loglevel, log_to_file, logfile):
    log_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(log_level, int):
        raise ValueError(f"Invalid log level: {loglevel}")
    logger_mosaic_fraction = logging.getLogger("MosaicFractionEstimator")
    logger_mosaic_fraction.setLevel(log_level)
    if log_to_file:
        logger_mosaic_fraction_file_handler = myutils.LockedFileHandler(
            logfile
        )
    else:
        logger_mosaic_fraction_file_handler = logging.StreamHandler()
    # logging.StreamHandler() 用于将日志消息发送到指定的流，如果没有指定流，则默认为标准错误流（sys.stderr）。这使得StreamHandler非常适用于将日志输出到控制台或标准输出/错误中，方便在开发过程中监视程序的行为。
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger_mosaic_fraction_file_handler.setFormatter(formatter)
    if not logger_mosaic_fraction.handlers:
        logger_mosaic_fraction.addHandler(logger_mosaic_fraction_file_handler)
    return logger_mosaic_fraction

def feature_extract_logger(loglevel, log_to_file, logfile):
    log_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(log_level, int):
        raise ValueError(f"Invalid log level: {loglevel}")
    logger_feature_extract = logging.getLogger("FeatureExtractor")
    logger_feature_extract.setLevel(log_level)
    if log_to_file:
        logger_feature_extract_file_handler = myutils.LockedFileHandler(
            logfile
        )
    else:
        logger_feature_extract_file_handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger_feature_extract_file_handler.setFormatter(formatter)
    if not logger_feature_extract.handlers:
        logger_feature_extract.addHandler(logger_feature_extract_file_handler)
    return logger_feature_extract
