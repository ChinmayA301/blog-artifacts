# MatrAIx persona input

Approach B uses the public MatrAIx Persona release as an optional persona pool.
The files are not vendored here.

## Public inputs

- [Dataset card](https://huggingface.co/datasets/MatrAIx2026/MatrAIx_Persona_1M_Public_Release)
- [Decoded 999-row sample](https://huggingface.co/datasets/MatrAIx2026/MatrAIx_Persona_1M_Public_Release/resolve/main/sample/sample.parquet)
- [1,290-field codebook](https://huggingface.co/datasets/MatrAIx2026/MatrAIx_Persona_1M_Public_Release/resolve/main/persona_codes.schema.json)

Download the small decoded sample for the conditioning audit:

```bash
curl -L \
  "https://huggingface.co/datasets/MatrAIx2026/MatrAIx_Persona_1M_Public_Release/resolve/main/sample/sample.parquet" \
  -o matraix-persona-sample.parquet
```

The sample inspected for this update contained 999 rows and seven provenance
labels. The release's 60/40 grounded/synthetic composition is a data-product
choice, not an estimate of any platform population. The dataset card also says
its marginal calibration does not guarantee a representative joint
distribution.

## Provenance rule

Even a human-grounded persona row becomes a **synthetic node attribute** when
assigned to an anonymous Higgs node. There is no identity join between the two
datasets. Results must therefore be described as:

> real graph topology, MatrAIx-derived synthetic node attributes, synthetic
> behavioral parameters, and a real optimization method.

Never describe the assigned personas as observed Higgs-user attributes.
