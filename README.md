# A Pipeline for Large-Scale Archaeological Site Detection: Integrating Remote Sensing Data with Historical Maps
This repository contains the code supporting the paper
Albrecht, R. (2027). A Pipeline for Large-Scale Archaeological Site Detection: Integrating Remote Sensing Data with Historical Maps. In: Dimitri, G.M., Aslan, S., Montanelli, S., Ravanelli, M., Subakan, C., Trentin, E. (eds) Artificial Neural Networks in Pattern Recognition. ANNPR 2026. Lecture Notes in Computer Science(), vol 16978. Springer, Cham. https://doi.org/10.1007/978-3-032-39028-8_34

## Repository structure
- `src/` contains all code files used for data preprocessing, training, inference, and evaluation.
- `hpc/` contains the SLURM scripts used to run training, validation, and inference on the HPC clusters.
- `Dockerfile` and `requirements.txt` define the container image and the Python dependencies.

## Environment
All code was run using docker and singularity containers. They can be built from the contained `Dockerfile` using the following commands:
```
docker save <container_id> -o tell.tar
apptainer build tell.sif docker-archive://tell.tar
```
The pinned Python dependencies are listed in `requirements.txt`.

## Data
The data used for training and all used historical maps are available here: https://zenodo.org/records/18307916 ([DOI](https://doi.org/10.5281/zenodo.18307915)). After downloading the data it should be placed into a `data/` directory at the root of the repository. The code expects the following layout:

```
data/
├── classification/
│   ├── classification_labels.csv   # tile labels; columns: choice, image, id ("Site" = positive)
│   ├── new_nosites.csv             # additional negative labels; columns: choice, image, id
│   ├── casini_site/                # positive tile images
│   └── nosite/                     # negative tile images
├── vadineanu/
│   ├── site/                       # external validation positives
│   └── nosite/                     # external validation negatives
├── maps/
│   ├── <map>.jpg                   # scanned historical maps
│   └── <map>_modified.tif          # georeferenced companion for each map
└── corona_images/
    └── <Country>/                  # Corona satellite scenes per country (*.tif)
```

## HPC scripts
The scripts in `hpc/` are SLURM batch scripts written for two different clusters:
- `meluxina.sh` and `meluxina_validate.sh` are for the Meluxina HPC.
- `unilu_hpc_predict.sh` is for the University of Luxembourg HPC.

Before submitting a job, replace the following placeholders in the scripts:
- `path/to/project/`: host path to this repository; it is mounted at `/tell_detection` inside the container.
- `path/to/data/`: host path to the downloaded `data/` directory.
- `path/to/results_dir/`: host path where the prediction results should be written.
- `/scratch/users/<username>/corona_images`: your scratch directory holding the Corona GeoTIFFs (University of Luxembourg HPC script).
- `<account_id>`: your project account on Meluxina.
- `<email>`: the email address used for SLURM job notifications.

The scripts assume that the `tell.sif` container image has been built (see Environment) and is available in the working directory.

### AI Usage
All comments in the repository and this README.md were generated using OpenCode 1.18.14 with the model DeepSeek V4 Flash.
