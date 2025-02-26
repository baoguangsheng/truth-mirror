from .detector_base import DetectorBase
from .baselines import Baselines
from .fast_detect_gpt import FastDetectGPT
from .binoculars import Binoculars
from .glimpse import Glimpse
from .radar import Radar
from .roberta import RoBERTa
from .detect_llm import LRR


def get_detector(name):
    name_detectors = {
        'roberta': ('roberta', RoBERTa),
        'radar': ('radar', Radar),
        'log_perplexity': ('log_perplexity', Baselines),
        'log_rank': ('log_rank', Baselines),
        'lrr': ('lrr', LRR),
        'fast_detect': ('fast_detect', FastDetectGPT),
        'glimpse': ('glimpse', Glimpse),
        'binoculars': ('binoculars', Binoculars),
    }
    if name in name_detectors:
        config_name, detector_class = name_detectors[name]
        return detector_class(config_name)
    else:
        raise NotImplementedError
