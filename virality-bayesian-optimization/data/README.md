# Data

These experiments run on the **SNAP Higgs Twitter dataset**, specifically the
retweet network, which is the diffusion substrate for influence spread.

## Source

Higgs Twitter Dataset — Stanford SNAP
https://snap.stanford.edu/data/higgs-twitter.html

Files (download the retweet edgelist at minimum):

| File | Used for |
|---|---|
| `higgs-retweet_network.edgelist.gz` | diffusion graph (required) |
| `higgs-social_network.edgelist.gz`  | full follow graph (not used in v1; ~54MB) |
| `higgs-mention_network.edgelist.gz` | optional alternative substrate |
| `higgs-reply_network.edgelist.gz`   | optional alternative substrate |
| `higgs-activity_time.txt.gz`         | temporal activity (future work) |

## Why these are not committed

The raw edgelists are large (the social network alone is ~54MB) and are
redistributed by SNAP under their own terms. They are intentionally
`.gitignore`d. Download them from the link above and place
`higgs-retweet_network.edgelist.gz` in the `pipeline/` directory, then run
`00_build_graph.py` to produce `rt_graph.pkl`.

## Citation

De Domenico, M., Lima, A., Mougel, P., & Musolesi, M. (2013).
The Anatomy of a Scientific Rumor. *Scientific Reports*, 3, 2980.
