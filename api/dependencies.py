from arch_a import ArchAProcessor
from arch_b import ArchBProcessor

_arch_a_processor = None
_arch_b_processor = None

# Зависимость для МЛ архитектуры
def get_arch_a_processor():
    global _arch_a_processor
    if _arch_a_processor is None:
        _arch_a_processor = ArchAProcessor()
    return _arch_a_processor

# Зависимость для ЛЛМ архитектуры
def get_arch_b_processor():
    global _arch_b_processor
    if _arch_b_processor is None:
        _arch_b_processor = ArchBProcessor()
    return _arch_b_processor