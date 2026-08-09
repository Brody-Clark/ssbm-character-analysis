# Super Smash Bros Melee Vision

Super Smash Bros Melee Vision (SSBMV) is a computer vision pipeline designed to detect, track, and match characters in Super Smash Bros Melee gameplay footage using classical image processing techniques written using OpenCV. This pipeline is designed to run on recorded footage and produces a game state that can be streamed to stdout or written to a json-lines (.jsonl) file.

## Table of Contents

- [Getting Started](#getting-started)
- [Command Line Options](#command-line-options)
- [Running with Test Videos](#running-with-test-videos)
- [Analyzing Results](#analyzing-results)

## Getting Started

This project is built using `OpenCV`, `scipy`, and `numpy`. These modules can be installed by running:

```bash
python -m pip install opencv-python numpy scipy scikit-image
```

Alternatively, the necessary modules can be installed using the supplied `requirements.txt`:

```bash
python -m pip install -r ./requirements.txt
```

**NOTE:** This project relies on pre-compiled templates for actor and HUD matching which are located in the `/data/templates/` directory. This directory should be
in the same root as the `/src/` directory to be detectable.

## Command Line Options


## Running with Test Videos

Three test recordings are provided and located in `\test\recordings` including `corneria.mp4`, `temple.mp4`, and `venom.mp4`.
To run the pipeline for a video, make sure the stage name matches the file name of the vide (e.g., `--stage corneria` for video file `corneria.mp4`).
Output can be chosen between `stdout` and a `jsonl` file. See [Command Line Arguments]() for more information. 

```bash
python src/ssbmv/main.py --input <path-to-video> --output "stdout" --stage <name-of-stage>
```

Detection is dependent on stage masking, however automated stage detection is beyond the scope of this project, so the stage name must be specified as a program argument.
Supported stage names include:

- temple
- corneria
- venom

**NOTE:** If using the provided test files, the stage name is the same as the file name without the extension.

Below is an example command to run the pipeline on the provided `temple.mp4` gameplay video and write the results to a json-lines file for analysis:
```bash
# Run the pipeline for the provided gameplay in the 'temple' stage
python ./src/ssbmv/main.py --input ./test/recordings/temple.mp4 --stage 'temple' --output file --file=./results.jsonl
```

## Analyzing Results

The results can be analyzed using the provided `analyzer.py` file located in the `/tools/` folder. The analysis requires a valid results file produced by the pipeline and the corresponding ground truth file for that video located in `./tests/labels/`.
Example:
```bash
# Analyze results for the 'temple.mp4' video against the ground truth
python ./tools/analyzer.py --results ./results.jsonl --ground-truth ./tests/labels/temple.json
```

This will provide the performance and accuracy results found in the supporting paper. 

