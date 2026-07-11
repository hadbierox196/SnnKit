# Data source

**File:** `aconnectome_white_1986_whole.csv`

**Source:** [openworm/ConnectomeToolbox](https://github.com/openworm/ConnectomeToolbox),
path `cect/data/aconnectome_white_1986_whole.csv`, fetched from
`main` branch via
`https://raw.githubusercontent.com/openworm/ConnectomeToolbox/main/cect/data/aconnectome_white_1986_whole.csv`.

**License:** MIT (per the `ConnectomeToolbox` repository's `LICENSE` file —
"All the code, data and models produced as part of the OpenWorm project
are open-source under the MIT licence.").

**Underlying dataset:** White, J.G., Southgate, E., Thomson, J.N., Brenner,
S. (1986). "The structure of the nervous system of the nematode
*Caenorhabditis elegans*." *Philosophical Transactions of the Royal
Society of London B*, 314(1165), 1-340. Distributed by the OpenWorm
project's Connectome Toolbox as a curated, structured CSV re-export of
the original dataset (see the toolbox's own citation/provenance notes at
<https://openworm.org/ConnectomeToolbox/> for the full chain of custody).

**Format:** tab-separated, columns `pre`, `post`, `type`
(`chemical`/`electrical`), `synapses` (synapse count for that pre/post
pair). 309 nodes (individually identified neurons, plus a small number of
aggregate non-neuronal targets such as `LegacyBodyWallMuscles` retained
from the original dataset), 2,961 edges.

**Not modified** from the fetched version, other than being vendored into
this repo for reproducibility (so `snnkit.connectome.loader` doesn't
depend on network access at import time). Re-fetch the URL above if you
want the latest version from upstream.
