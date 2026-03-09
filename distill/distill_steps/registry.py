from importlib import import_module
from functools import partial

def Registry(distill_cfg):
    # module, class_name = REGISTRY[name].rsplit(".", 1)
    # return getattr(import_module(module), class_name)
    module = import_module(REGISTRY[distill_cfg.type])
    distill_step = getattr(module, "distill_step")
    distill_step = partial(distill_step, **distill_cfg)
    return distill_step



REGISTRY = {

    # Distillation steps
    "supervised": "distill.distill_steps.supervised",
    "hstates": "distill.distill_steps.hstates",
    "matrices": "distill.distill_steps.matrices",
    "supervised_instruct": "distill.distill_steps.supervised_instruct",
    "dpo": "distill.distill_steps.dpo",

}
