Run the single-region search with Python 3.11.

```sh
python3 finder.py --scen DIR --out OUT --budget-s 1800
```

The search reads the DECam catalogues, observer positions and exposure metadata in DIR. It writes ranked linked detections to OUT/candidates.json and checkpoints during the search. The CPU implementation supports the same inputs. CUDA accelerates the large search and verification batches when available.
