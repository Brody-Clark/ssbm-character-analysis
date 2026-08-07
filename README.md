# Super Smash Bros Melee Vision

Super Smash Bros Melee Vision is a computer vision pipeline designed to detect, track, and match characters in Super Smash Bros Melee gameplay footage using classical image processing techniques written using OpenCV. This pipeline is designed to run on recorded footage and produces a game state that can be streamed to stdout or written to a
json-lines (.jsonl) file.

## Getting Started

This project is built using OpenCV, scipy, and numpy. The necessary modules can be installed using the supplied requirements.txt.

```bash
pip install -r ./requirements.txt
```

This project relies on pre-compiled templates for actor and HUD matching which are located in the `/data/templates/` directory. This directory should be
in the same root as the `/src/` directory to be detectable.

## Running With Test Videos

The test recordings are located in `\test\recordings` and include 3 videos named `corneria.mp4`, `temple.mp4`, and `venom.mp4`.
To run the pipeline for a video, make sure the stage name matches the file name of the vide (e.g., --stage='corneria' for video file 'corneria.mp4').
Output can be chosen between stdout and a jsonl file. See [Command Line Arguments]() for more information. 

```bash
python src/ssbmv/main.py --input <path-to-video> --output "stdout" --stage <name-of-stage>
```

Detection is dependent on stage masking, however automated stage detection is beyond the scope of this project, so the stage name must be specified as a program argument.
Supported stage names include:

- temple
- corneria
- venom

[!NOTE] If using the provided test files, the stage name is the same as the file name without the extension.

Below is an example command to run the pipeline on the provided `temple.mp4` gameplay video and write the results to a json-lines file for analysis:
```bash
python ./src/ssbmv/main.py --input ./test/recordings/temple.mp4 --stage 'temple' --output file --file=./results.jsonl
```

The results can be analyzed using the provided `analyzer.py` file. Example:
```bash
python ./tools/analyzer.py --input=results.jsonl
```

This will provide the performance and accuracy results found in the supporting paper. 