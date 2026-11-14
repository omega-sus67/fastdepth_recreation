# FastDepth checkpoints and provenance

The four original-author download URLs still time out over both HTTP and HTTPS. Exact attempts are recorded in `results/checkpoints/original-source-attempts.json`; the mirror search is recorded in `results/checkpoints/source-search.json`.

## Available community pruned candidate

- Legacy file: `candidates/beihaixingchen-best_result.pth.tar` (11,038,470 bytes).
- Immutable source: [upstream fork upload](https://github.com/beihaixingchen/fast-depth/blob/bca2e09002f4e214f76b024881988967b8561865/best_result.pth.tar).
- SHA-256: `6bcaed246a9424f3c9cf16e554529a87e98cc72c3e93080ce3f77631974665b0`.
- Converted file: `converted/community-skipadd-pruned.pth`.
- Explicit architecture: `converted/community-skipadd-pruned.architecture.json`.
- Model: pruned MobileNetSkipAdd, 1,360,786 parameters, `1 × 3 × 224 × 224` input and `1 × 1 × 224 × 224` output.
- Source status: **community mirror; not authenticated by an original-author checksum**. The upstream source license is preserved under `third_party/fast-depth/LICENSE`; a separate checkpoint-specific license/authenticity statement was not established.

The converted model matches the legacy forward output exactly on the recorded synthetic check and reloads bit-exactly. On the vendored upstream RGB example, its output agrees with the bundled TVM example prediction within `rtol=1e-3, atol=1e-3`. This is useful supporting evidence, not authentication or a 654-image accuracy result. Legacy saved metrics are metadata claims, not remeasured project results.

The unpruned NNConv5, depthwise NNConv5, and depthwise skip-add architectures work with explicitly labeled random weights. Their original pretrained checkpoints are still missing. Never use random weights as reproduction results or replace a pruned model with an unpruned architecture silently.

## Reproduce acquisition and conversion

```bash
# Expected to report unavailable until the original server recovers:
bash scripts/python.sh scripts/download_checkpoints.py

# Retrieve/verify the checksum-pinned community candidate:
bash scripts/python.sh scripts/download_checkpoints.py --community-pruned

# Conversion preserves the legacy file; choose a NEW output filename:
bash scripts/python.sh scripts/convert_checkpoint.py checkpoints/candidates/beihaixingchen-best_result.pth.tar \
  --sha256 6bcaed246a9424f3c9cf16e554529a87e98cc72c3e93080ce3f77631974665b0 \
  --variant mobilenet-nnconv5dw-skipadd-pruned \
  --source-url https://raw.githubusercontent.com/beihaixingchen/fast-depth/bca2e09002f4e214f76b024881988967b8561865/best_result.pth.tar \
  --source-status community_mirror_unverified \
  --output checkpoints/converted/community-pruned-reconverted.pth \
  --report results/checkpoints/reconversion.json

bash scripts/python.sh scripts/check_models.py --structural --checkpoint checkpoints/converted/community-skipadd-pruned.pth
```

The legacy adapter uses an explicit class/function allowlist and requires the expected SHA-256. PyTorch 0.4's persistent module IDs are incompatible with modern `weights_only=True`, so conversion supplies a restricted unpickler rather than unrestricted pickle. Converted files use ordinary explicit config/state dictionaries and load with `weights_only=True`. Unknown globals/types and mismatched hashes fail. No source from the community fork is executed.
