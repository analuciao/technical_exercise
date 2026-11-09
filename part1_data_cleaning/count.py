"""Print every count to report in the README and summary, from file_check.py's outputs."""
from pathlib import Path

import pandas as pd
import pydicom

RELEASE_DIR = Path("/Users/analuciaochoa/Downloads/release")
OUT_DIR = Path("/Users/analuciaochoa/technical_exercise/part1_data_cleaning/outputs")

# key phrase in file_check.csv's Reasoning column -> label to print
REASONS = {
    "Can't be opened as DICOM":            "Not a valid DICOM|Could not read file",
    "No image / image can't be decoded":   "No image data|could not be decoded",
    "Fingerprint mismatch":                "Fingerprint mismatch",
    "Not listed in SHA256SUMS.txt":        "Not listed in SHA256SUMS",
    "Not listed in echo-record-list.csv":  "Not listed in echo-record-list",
    "Duplicate of another file":           "Duplicate of",
    "Single image, not a video":           "Single image",
    "Not an ultrasound":                   "Not an ultrasound",
    "Too short for one heartbeat":         "Too short",
}


def header(title):
    print(f"\n{title}\n{'-' * len(title)}")


def file_counts():
    df = pd.read_csv(OUT_DIR / "file_check.csv")
    reasons = df["Reasoning"].fillna("")
    missing = reasons.str.contains("missing from folder", case=False)
    in_folder = df[~missing]
    usable = in_folder[in_folder["Usability"] == "Usable"]

    header("Files")
    print(f"Files in release folder:              {len(in_folder)}")
    print(f"Usable videos:                        {len(usable)}")
    print(f"Not usable:                           {(in_folder['Usability'] == 'Not Usable').sum()}")
    print(f"Listed in record list but missing:    {missing.sum()}")

    header("Reasons for Not Usable (a file can have several)")
    for label, pattern in REASONS.items():
        print(f"{label:38} {reasons[~missing].str.contains(pattern, case=False).sum()}")

    return usable["File Name"].tolist()


def patient_count(usable_paths):
    """PatientID equals subject_id in this release, so count distinct PatientIDs in the originals."""
    ids = set()
    for rel_path in usable_paths:
        ds = pydicom.dcmread(RELEASE_DIR / rel_path, stop_before_pixels=True)
        pid = str(ds.get("PatientID", "")).strip()
        if pid:
            ids.add(pid)
    header("Patients")
    print(f"Distinct patients (subject_id) among usable videos: {len(ids)}")


def phi_counts():
    df = pd.read_csv(OUT_DIR / "phi_found.csv")
    df["Reason"] = df["Reason"].fillna("(no reason)")

    header("PHI findings by reason")
    summary = df.groupby("Reason").agg(findings=("Field", "size"), files=("File Name", "nunique"))
    print(summary.sort_values("findings", ascending=False).to_string())
    print(f"\nFiles with at least one finding: {df['File Name'].nunique()}")

    header("Most common fields flagged")
    print(df["Field"].value_counts().head(15).to_string())

    dates = df[df["Reason"].str.contains("date", case=False)]
    if len(dates):
        header("Unshifted dates by field")
        print(dates["Field"].value_counts().to_string())

    free_text = df[df["Reason"].str.contains("Free text", case=False)]
    if len(free_text):
        header("Free-text fields containing a name or ID")
        print(free_text["Field"].value_counts().to_string())

    old = df[df["Reason"].str.contains("90", case=False)]
    header("Patients aged 90+")
    print(f"Files with PatientAge >= 90: {old['File Name'].nunique()}")


if __name__ == "__main__":
    usable_paths = file_counts()
    patient_count(usable_paths)
    phi_counts()
    print("\nVerification result (X/X passing) is printed at the end of file_check.py.")