Technical Exercise

Two independent parts, each in its own folder with its own README:

1. Data cleaning (part1_data_cleaning/):
   Sorts a release of echo DICOM files into usable / not usable, finds and removes patient information (header fields and names burned into the images), and writes cleaned copies.
2. LV segmentation (part2_lv_segmentation/):
   Trains a U-Net on EchoNet-Dynamic expert tracings to outline the left ventricle, then compares predicted outlines, areas and fractional area change against the experts.

Setup:
Python 3.11+ on macOS (Apple Silicon GPU via PyTorch mps is used if available).

bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
Data

No data is included in this repository. Each part's README says where its data is expected and how to regenerate the output. Files containing patient information (phi_found.csv, original and cleaned DICOMs) are excluded via .gitignore.