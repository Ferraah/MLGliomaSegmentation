# MLGliomaSegmentation

Running on iris/aion:
```
salloc --ntasks-per-node 128 -c 1 
micromamba install monai kagglehub nibabel
python src/main.py
```
Remember to delete the file split_indices whenever there is the need to have a new indices split. 
