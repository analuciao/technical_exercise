from pathlib import Path
 
# paths 
ECHONET_DIR = Path("/Users/analuciaochoa/Downloads/EchoNet-Dynamic")
VIDEO_DIR = ECHONET_DIR / "Videos"
FILELIST_PATH = ECHONET_DIR / "FileList.csv"
TRACINGS_PATH = ECHONET_DIR / "VolumeTracings.csv"
 
PART2_DIR = Path("/Users/analuciaochoa/technical_exercise/part2_lv_segmentation")
DATA_DIR = PART2_DIR / "data"          # generated .npy files
OUTPUT_DIR = PART2_DIR / "outputs"     # images, CSVs
MODEL_PATH = DATA_DIR / "best_model.pt"
 
# data 
LIMITS = {"TRAIN": None, "VAL": None, "TEST": None}   # videos per split
PLOT_CHECK = True                                # save an overlay grid of expert masks
 
# model 
ENCODER = "resnet34"
IMG_SIZE = 128             # divisible by 32 for the U-Net
 
# training
EPOCHS = 30
PATIENCE = 5
BATCH_SIZE = 16
LR = 1e-4
 
# prediction 
THRESHOLD = 0.5            
EVAL_SPLIT = "TEST"       