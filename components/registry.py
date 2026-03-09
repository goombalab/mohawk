from importlib import import_module

def Registry(name):
    module, class_name = REGISTRY[name].rsplit(".", 1)
    return getattr(import_module(module), class_name)



REGISTRY = {

    # LMHeads
    "mixer_seq_simple": "components.LMHeads.mixer_seq_simple.LMHeadModel",
    "LayeredMambaLM": "components.LMHeads.LayeredMambaLM.LMHeadModel",
    "LlamaForCausalLM": "components.LMHeads.LlamaForCausalLM.LlamaForCausalLM",


    # MixerModels
    "LlamaModel": "components.MixerModels.LlamaModel.MixerModel",
    "Qwen2Model": "components.MixerModels.Qwen2Model.MixerModel",
    "MixerModelRotary": "components.MixerModels.MixerModelRotary.MixerModel",


    # Blocks
    "OriginalBlock": "components.blocks.OriginalBlock.Block",
    "MambaPhi": "components.blocks.MambaPhi.Block",
    "LlamaBlock": "components.blocks.LlamaBlock.Block",
    "Qwen2Block": "components.blocks.Qwen2Block.Block",
    "FalconBlock": "components.blocks.FalconBlock.Block",
    "DoubleBlockMerger": "components.blocks.DoubleBlock.DoubleBlockMerger",
    "DoubleBlockAdapter": "components.blocks.DoubleBlock.DoubleBlockAdapter",
    "DoubleBlockHymba": "components.blocks.DoubleBlock.DoubleBlockHymba",
    "DoubleBlockVanilla": "components.blocks.DoubleBlock.DoubleBlockVanilla",


    # Cores
    "Mamba1": "mamba_ssm.Mamba",
    "Mamba2": "mamba_ssm.Mamba2",
    "DiscreteMamba2": "components.cores.discrete_mamba2.Mixer",
    "Qwen2Attention": "components.cores.Qwen2Attention.Mixer",
    "LlamaAttention": "components.cores.LlamaAttention.Mixer",
    "PhiAttention": "components.cores.phi_attention.Mixer",
    "DiscreteMamba2Rotary": "components.cores.discrete_mamba2_rotary.Mixer",
    "FalconMambaMixer": "transformers.models.falcon_mamba.modeling_falcon_mamba.FalconMambaMixer",

}
