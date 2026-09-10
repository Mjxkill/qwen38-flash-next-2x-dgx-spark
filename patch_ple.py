from pathlib import Path

p = Path("/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/ple_layer.py")
src = p.read_text()

old = """    if not isinstance(quant_config, Fp8Config):
        return None
"""
new = """    if not isinstance(quant_config, Fp8Config):
        # vLLM #54765 (fix upstream incomplet) : le checkpoint RadixArk NVFP4
        # quantifie bien la table PLE en FP8 (tenseur ngram_embedding.weight_scale
        # présent) alors que sa quantization_config la déclare dans "ignore".
        # Le fix amont ne couvre que ModelOptMixedPrecisionConfig ; pour un
        # ModelOptNvFp4Config on retombe sur l'embedding nu -> pas de
        # weight_scale enregistré -> ValueError au chargement du dernier shard.
        # Qwen4ExpPLEFp8EmbeddingMethod ne lit aucun champ de quant_config.
        name = quant_config.get_name() if quant_config is not None else ""
        if name in ("modelopt_fp4", "modelopt"):
            return Qwen4ExpPLEFp8EmbeddingMethod()
        return None
"""

if "vLLM #54765" in src:
    raise SystemExit("déjà patché")
if src.count(old) != 1:
    raise SystemExit(f"motif introuvable ou ambigu ({src.count(old)}) — source dérivée")
p.write_text(src.replace(old, new))
print("PATCHED", p)
