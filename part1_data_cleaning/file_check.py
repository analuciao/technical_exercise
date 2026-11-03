import csv
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import pydicom
from pydicom.errors import InvalidDicomError

RELEASE_DIR = Path("/Users/analuciaochoa/Downloads/release")
FILES_DIR = RELEASE_DIR / "files"
SUMS_PATH = RELEASE_DIR / "SHA256SUMS.txt"
ECHO_LIST_PATH = RELEASE_DIR / "echo-record-list.csv"
OUT_DIR = Path("/Users/analuciaochoa/technical_exercise/part1_data_cleaning/outputs")
CLEAN_DIR = OUT_DIR / "cleaned"
PREVIEW_DIR = OUT_DIR / "previews"

MIN_SECONDS = 1.0          # one heartbeat at 60 bpm
MIN_FRAMES_FALLBACK = 40   # only used when the header has no FrameTime

# --- step 2 settings ---
CURRENT_YEAR = 2026        # shifted dates are in the future (~2150); anything <= this is real
PHI_KEYWORDS = {
    "PatientName", "PatientID", "OtherPatientIDs", "PatientBirthDate", "PatientBirthTime",
    "AccessionNumber", "PatientAddress", "PatientTelephoneNumbers",
    "InstitutionName", "InstitutionAddress",
}
TEXT_VRS = {"LO", "SH", "ST", "LT", "UT"}

# Burned-in names: look at the images in PREVIEW_DIR, then fill these in.
BURN_IN_BOXES = [(0, 43, 0, 50), (0, 23, 0,65) 
       ]       # (y_start, y_end, x_start, x_end) of the region to black out
BURNED_IN_FILES = set()    # rel paths with a printed name, e.g. {"files/abc.dcm"}


@dataclass
class FileResult:
    name: str
    usability: str = "Usable"
    reasoning: list[str] = field(default_factory=list)

    def fail(self, reason):
        self.usability = "Not Usable"
        self.reasoning.append(reason)


# ---------- loading reference files ----------

def load_expected_checksums(path):
    expected = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        checksum, name = line.split(maxsplit=1)
        expected[Path(name.lstrip("*")).name] = checksum   # sha256sum may prefix names with *
    return expected


def load_echo_list(path):
    with open(path, newline="", encoding="utf-8") as f:
        return {row["dicom_filepath"] for row in csv.DictReader(f)}


# ---------- step 1 checks ----------

def sha256(file_path):
    with open(file_path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def check_checksum(file_path, actual, expected, result):
    listed = expected.get(file_path.name)
    if listed is None:
        result.fail("Not listed in SHA256SUMS.txt")
    elif listed != actual:
        result.fail("Fingerprint mismatch")


def check_duplicate(actual, seen_hashes, result):
    if actual in seen_hashes:
        result.fail(f"Duplicate of {seen_hashes[actual]}")
    else:
        seen_hashes[actual] = result.name


def read_dicom(file_path, result):
    """Return the dataset, or None if the file can't be read as DICOM."""
    try:
        return pydicom.dcmread(file_path)
    except InvalidDicomError:
        result.fail("Not a valid DICOM file")
    except Exception as e:
        result.fail(f"Could not read file ({type(e).__name__}: {e})")
    return None


def check_pixels(ds, result):
    """dcmread is lazy: the header can load while the image is missing or broken."""
    if "PixelData" not in ds:
        result.fail("No image data in file")
        return False
    try:
        ds.pixel_array
    except Exception as e:
        result.fail(f"Image data could not be decoded ({type(e).__name__})")
        return False
    return True


def check_is_video(ds, result):
    num_frames = int(ds.get("NumberOfFrames") or 1)
    modality = ds.get("Modality")
    if num_frames <= 1:
        sop = ds.get("SOPClassUID")
        kind = sop.name if sop else "unknown type"
        result.fail(f"Single image, not a video ({kind})")
    elif modality and modality != "US":
        result.fail(f"Not an ultrasound (Modality={modality})")
    return num_frames


def check_heartbeat(ds, num_frames, result):
    frame_time = ds.get("FrameTime")             # milliseconds per frame
    if frame_time:
        seconds = num_frames * float(frame_time) / 1000
        if seconds < MIN_SECONDS:
            result.fail(f"Too short for one heartbeat: {seconds:.2f}s of video")
    elif num_frames < MIN_FRAMES_FALLBACK:
        result.fail(f"Too short for one heartbeat: only {num_frames} frames")


# ---------- step 2: finding PHI ----------

def as_text(value):
    """Private tags often hold raw bytes; turn any value into searchable uppercase text."""
    if isinstance(value, bytes):
        value = value.decode("latin-1", errors="ignore")
    return str(value).strip().upper()


def collect_identifiers(ds):
    """First pass: gather this file's names and IDs so we can search free text for them."""
    known = set()
    for elem in ds.iterall():
        if elem.is_empty:
            continue
        text = as_text(elem.value)
        is_name = elem.VR == "PN" or (elem.tag.is_private and "^" in text)   # DICOM names look like LAST^FIRST
        if is_name or elem.keyword in ("PatientID", "OtherPatientIDs"):
            known.add(text)
            known.update(part for part in re.split(r"[\^\s]+", text) if len(part) >= 2)
    known.discard("")
    return known


def is_real_date(value):
    year = as_text(value)[:4]
    return year.isdigit() and int(year) <= CURRENT_YEAR


def contains_known(value, known):
    text = as_text(value)
    return any(re.search(rf"\b{re.escape(k)}\b", text) for k in known)


def find_reason(elem, known):
    """Return why this element is PHI, or None if it isn't."""
    if elem.tag.is_private:
        return "Private (vendor-specific) tag"
    if elem.keyword in PHI_KEYWORDS:
        return "Direct identifier"
    if elem.VR == "PN":
        return "Person name"
    if elem.VR in ("DA", "DT") and is_real_date(elem.value):
        return "Unshifted (real) date"
    if elem.VR in TEXT_VRS and contains_known(elem.value, known):
        return "Free text contains a name or ID"
    return None


def scan_header(ds, known):
    """Second pass: list every PHI finding. Detection only, does not change ds."""
    findings = []
    for elem in ds.iterall():
        if elem.is_empty or elem.tag.is_private_creator:
            continue
        reason = find_reason(elem, known)
        if reason:
            value = as_text(elem.value)
            findings.append([str(elem.tag), elem.keyword or elem.name, elem.VR,
                             value[:60] + ("..." if len(value) > 60 else ""), reason])
    if ds.get("BurnedInAnnotation") == "YES":
        findings.append(["(0028,0301)", "BurnedInAnnotation", "CS", "YES",
                         "Header says text is printed on the image"])
    return findings


# ---------- step 2: cleaning ----------

def save_preview(ds, out_path):
    """Save the first frame as a PNG so you can look for printed names."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame = ds.pixel_array[0]
    plt.imsave(out_path, frame, cmap=None if frame.ndim == 3 else "gray")


def black_out(ds, boxes):
    """Paint the box black on every frame."""
    arr = ds.pixel_array.copy()
    for y0, y1, x0, x1 in boxes:
        arr[:, y0:y1, x0:x1] = 0     # works for gray (F,H,W) and color (F,H,W,3)
        if ds.SamplesPerPixel == 3:
            arr[:, y0:y1, x0:x1, :] = 0
        photometric = "RGB"          # pixel_array returns RGB for color images
    else:
        arr[:, y0:y1, x0:x1] = 0
        photometric = ds.PhotometricInterpretation
    ds.set_pixel_data(arr, photometric, ds.BitsStored)   # pydicom 3: rewrites PixelData uncompressed
    ds.BurnedInAnnotation = "NO"


def clean(ds, known, pseudonym, blackout_box):
    """Edit ds in place: replace or blank every PHI field, drop private tags."""
    for elem in ds.iterall():
        if elem.is_empty or elem.tag.is_private:
            continue
        if find_reason(elem, known) is None:
            continue
        if elem.keyword in ("PatientID", "PatientName"):
            elem.value = pseudonym     # keeps a patient's files linked without the real ID
        else:
            elem.value = ""            # blank rather than delete, so the file stays valid DICOM
    ds.remove_private_tags()
    if blackout_box:
        black_out(ds, blackout_box)
    ds.PatientIdentityRemoved = "YES"
    ds.DeidentificationMethod = "Names/IDs replaced, real dates blanked, private tags removed"


def verify(cleaned_path, known):
    """Reopen the cleaned file and list any known name or ID still in the header."""
    ds = pydicom.dcmread(cleaned_path)
    leaks = []
    for elem in ds.iterall():
        if elem.keyword == "PixelData" or elem.is_empty:
            continue
        if elem.tag.is_private:
            leaks.append(f"private tag {elem.tag} still present")
        elif contains_known(elem.value, known):
            leaks.append(f"{elem.keyword or elem.name}: {as_text(elem.value)[:40]}")
    return leaks


# ---------- main ----------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    expected = load_expected_checksums(SUMS_PATH)
    echo_list = load_echo_list(ECHO_LIST_PATH)

    seen_hashes = {}      # content hash -> first file with that content
    files_present = set()
    usable_count = 0
    pseudonyms = {}       # real PatientID -> PATIENT_001, ...
    verify_failures = 0

    with open(OUT_DIR / "file_check.csv", "w", newline="", encoding="utf-8") as check_f, \
         open(OUT_DIR / "phi_found.csv", "w", newline="", encoding="utf-8") as phi_f:
        check_writer = csv.writer(check_f)
        phi_writer = csv.writer(phi_f)
        check_writer.writerow(["File Name", "Usability", "Reasoning"])
        phi_writer.writerow(["File Name", "Tag", "Field", "VR", "Value", "Reason"])

        for file_path in sorted(FILES_DIR.rglob("*")):
            if not file_path.is_file() or file_path.name.startswith("."):
                continue

            rel_path = file_path.relative_to(RELEASE_DIR).as_posix()   # e.g. "files/abc.dcm"
            files_present.add(rel_path)
            result = FileResult(name=rel_path)

            # --- step 1 ---
            actual = sha256(file_path)
            check_checksum(file_path, actual, expected, result)
            check_duplicate(actual, seen_hashes, result)
            if rel_path not in echo_list:
                result.fail("Not listed in echo-record-list.csv")

            ds = read_dicom(file_path, result)
            if ds is not None and check_pixels(ds, result):
                num_frames = check_is_video(ds, result)
                if num_frames > 1:
                    check_heartbeat(ds, num_frames, result)

            check_writer.writerow([result.name, result.usability, "; ".join(result.reasoning) or "N/A"])

            if ds is None:
                continue

            # --- step 2: find PHI in every readable file ---
            known = collect_identifiers(ds)
            for finding in scan_header(ds, known):
                phi_writer.writerow([rel_path, *finding])

            # --- step 2: clean usable files only ---
            if result.usability != "Usable":
                continue
            usable_count += 1

            real_id = as_text(ds.get("PatientID", "")) or "UNKNOWN"
            pseudonym = pseudonyms.setdefault(real_id, f"PATIENT_{len(pseudonyms) + 1:03d}")

            save_preview(ds, PREVIEW_DIR / "before" / f"{file_path.stem}.png")
            needs_blackout = BURN_IN_BOXES is not None
            clean(ds, known, pseudonym, BURN_IN_BOXES if needs_blackout else None)
            if needs_blackout:
                save_preview(ds, PREVIEW_DIR / "after" / f"{file_path.stem}.png")

            cleaned_path = CLEAN_DIR / rel_path
            cleaned_path.parent.mkdir(parents=True, exist_ok=True)
            ds.save_as(cleaned_path)

            leaks = verify(cleaned_path, known)
            if leaks:
                verify_failures += 1
                print(f"VERIFY FAILED {rel_path}: {leaks}")

        # listed in echo-record-list.csv but not in the folder
        missing = sorted(echo_list - files_present)
        for rel_path in missing:
            check_writer.writerow([rel_path, "Not Usable", "Listed in echo-record-list.csv but missing from folder"])

    print(f"Missing files: {len(missing)}")
    print(f"Usable videos: {usable_count}")
    print(f"Distinct patients among usable videos: {len(pseudonyms)}")
    print(f"Cleaned files passing verification: {usable_count - verify_failures}/{usable_count}")


if __name__ == "__main__":
    main()