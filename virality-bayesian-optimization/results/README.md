# Approach B results status

`persona_conditioning_sample_audit.json` is a reproducible schema and
conditioning audit of the 999-row decoded public MatrAIx sample. It is not a
result from the 223,833-node Higgs experiment.

The audit found:

- 665 of 999 records pass the declared age floor;
- the effective sample size after soft platform-fit conditioning is 652.8;
- field coverage ranges from 18.0% (`ind_sports`) to 77.0%
  (`trait_curiosity`) across the requested model fields;
- the derived ML-paper and sports-clip affinity signals are distinct rather
  than duplicates (sample correlation: -0.225);
- source composition is mixed: 395 synthetic rows and 604 rows derived from
  several grounded sources.

These checks support a **synthetic sensitivity experiment**. They do not make
the sampled personas representative of Twitter in 2012, and there is no
observed join between a persona and a Higgs graph node. For that reason, the
full output file `persona_conditioned_bo.json` is intentionally absent until
the cached Higgs graph is supplied and all three assignment couplings are run.

The input file is pinned in the JSON by SHA-256 so the audit can be reproduced
against the same public sample.
