# GraphEmbeddings
This repository contains realization of `DDoS` (aka `Histogram loss`)
algorithm for generating graph embeddings.


## How to run
install dependencies from requirements.txt and launch the run script

`python run.py`

You can also change parameters.

```
python run.py --dimensions 32
python run.py --dataset polbooks
```
 
Look at run.py source for all the params.


## Structure
All code is stored in folder `final_src`

Realization of several embedding algorithm including our algorithm
can be found in folder `transformers`.

Folder `io_utils` contains code responsible for reading
and writing graphs and embeddings.

Folder `transformation` contains generic code to generate an embedding
with any available algorithm.

Other folders represent sets of experiments for comparing algorithms:
`link_prediction`, `classification` and `clusterization`.

