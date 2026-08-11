# Super Smash Bros Melee Vision

Super Smash Bros Melee Vision (SSBMV) is a computer vision pipeline designed to detect, track, and match characters in Super Smash Bros Melee gameplay footage using classical image processing techniques written using OpenCV. This pipeline is designed to run on recorded footage and produces a game state that can be streamed to stdout or written to a json-lines (.jsonl) file.

## Table of Contents

- [Super Smash Bros Melee Vision](#super-smash-bros-melee-vision)
  - [Table of Contents](#table-of-contents)
  - [Prerequisites](#prerequisites)
  - [Command Line Options](#command-line-options)
    - [Options](#options)
  - [Running with Test Videos](#running-with-test-videos)
    - [Quick Start Example](#quick-start-example)
    - [Visualizing with Debug](#visualizing-with-debug)
  - [Analyzing Results](#analyzing-results)

## Prerequisites

This program is built with Python, to install python download the latest version of Python 3 from `https://www.python.org/`.

**Python Version:** Python 3.10 or higher recommended.

This project uses using `OpenCV`, `scipy`, `scikit-image`, and `numpy`. These modules can be installed directly by running:

```bash
python -m pip install opencv-python numpy scipy scikit-image
```

or by using the supplied `requirements.txt`:

```bash
python -m pip install -r ./requirements.txt
```

**NOTE:** This project relies on pre-compiled templates for actor and HUD matching which are located in the `/data/templates/` directory. This directory should be
in the same root as the `/src/` directory to be detectable.

## Command Line Options

```bash
python ./src/main.py -i <path_to_video> -s <stage> [--output {stdout,file}] [--file <output_file>] [--debug]

```

### Options

| Argument | Short | Required | Type | Default | Choices / Format | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `--input` | `-i` | **Yes** | `str` | *None* | File path | Path to the input video file to analyze. Relative paths are resolved against the project root. |
| `--stage` | `-s` | **Yes** | `str` | *None* | `temple`, `corneria`, `venom` | Name of the stage to use for analysis. |
| `--output` |  | No | `str` | `stdout` | `stdout`, `file` | Destination for the analysis output stream. |
| `--file` |  | Conditional | `str` | *None* | Must end in `.jsonl` | File path for output destination. **Required** if `--output` is set to `file`. |
| `--debug` |  | No | Flag | `False` | N/A | Enables debug mode during pipeline processing. Video is advanced by pressing the `Space` key. |

## Running with Test Videos

Three test recordings are provided and located in `\test\recordings` including `corneria.mp4`, `temple.mp4`, and `venom.mp4`.
To run the pipeline for a video, make sure the stage name matches the file name of the video (e.g., `--stage corneria` for video file `corneria.mp4`).
Output can be chosen between `stdout` and a `jsonl` file. See [Command Line Options](#command-line-options) for more information.

```bash
python ./src/main.py --input <path-to-video> --output "stdout" --stage <name-of-stage>
```

Detection is dependent on stage-specific masking, however automated stage detection is beyond the scope of this project, so the stage name must be specified as a program argument.
Supported stage names include:

- temple
- corneria
- venom

### Quick Start Example

Below is an example command to run the pipeline on the provided `temple.mp4` gameplay video and print the results to the console:
```bash
# From the root directory, run the pipeline for the provided gameplay in the 'temple' stage
python ./src/main.py --input ./test/recordings/temple.mp4 --stage temple
```

Below is an example command to run the pipeline on the provided `temple.mp4` gameplay video and write the results to a json-lines file for analysis:
```bash
# From the root directory, run the pipeline for the provided gameplay in the 'temple' stage
python ./src/main.py --input ./test/recordings/temple.mp4 --stage temple --output file --file=./results.jsonl
```

### Visualizing with Debug

The `--debug` flag allows you to run the pipeline with debug visuals enabled which display a window with characters and HUD elements
visualized.

**NOTE:** Frames can be advanced by clicking the `Space` key:

```bash
# Visualize results of game state prediction. Press Spacebar to advance frames
python ./src/main.py -i path/to/match.mp4 -s <stage-name> --debug

```

## Analyzing Results

Logged results can be analyzed using the provided `analyzer.py` file located in the `/tools/` folder. The analysis requires a valid results file produced by the pipeline and the corresponding ground truth file for that video located in `./tests/labels/`.
Example:
```bash
# Analyze results.jsonl produced for the 'temple.mp4' video against the ground truth
python ./tools/analyzer.py --results ./results.jsonl --ground-truth ./tests/labels/temple.json
```

This will print the performance and accuracy metrics found in the supporting paper to the console.
