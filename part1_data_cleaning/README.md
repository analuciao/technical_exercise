Part 1: Sorting and de-identifying echo DICOM files

file_check.py goes through every file in the release, decides whether it is usable, records any patient information it finds, and writes a cleaned copy of each usable file.

How to run

Expected input (paths set at the top of file_check.py):

release/
    files/                  # the DICOM files
    SHA256SUMS.txt
    echo-record-list.csv
bash
python file_check.py

Outputs go to outputs/:

Output:
file_check.csv:	One row per file: Usable / Not Usable and the reasons. Also one row per file listed in echo-record-list.csv but missing from the folder.
phi_found.csv:	One row per finding: file, DICOM tag, field, VR, value, and why it was flagged. Contains patient information; not committed to the repo.
cleaned/:	De-identified copies of the usable files only
previews/before, previews/after:	First frame of each usable file before and after blacking out burned-in text.

Step 1: Sorting the files

Every file in files/ is checked, regardless of extension, since names can be misleading. Hidden system files (e.g. .DS_Store) are skipped. A file is Not Usable if any check fails; all failing reasons are listed.

Check:
Opens as DICOM:	pydicom.dcmread. The image is also decoded, because dcmread only reads the header: a file can open but have missing or corrupt pixel data. Unexpected errors are recorded with their message.
Fingerprint:	SHA-256 of the file compared with SHA256SUMS.txt. Files absent from the list are flagged separately from mismatches.
On the record list	Path compared with echo-record-list.csv; listed files missing from the folder get their own rows.
Duplicate:	Same SHA-256 as an earlier file (same content under a different name). The later copy is flagged as Duplicate of <first file>.
Heart video:	More than one frame and Modality = US. Single images are reported with their SOP Class (e.g. Secondary Capture).
One full heartbeat	Video length = frames × FrameTime must be at least 1 second (one beat at 60 bpm). If FrameTime is missing, a fallback of 40 frames is used.

Step 2: Removing patient information
What counts as PHI

Each header element gets at most one reason:

Rule:
Private (vendor-specific) tag:	Real names were found here; all private tags are removed.
Direct identifier:	PatientName, PatientID (the MRN), OtherPatientIDs, PatientBirthDate, AccessionNumber, institution name/address.
Any person-name (PN) field:	Operator and physician names.
Unshifted date:	Dates in this release were shifted into the future (~2150), so any DA/DT date with a year ≤ 2026 is treated as a real date.
Free text containing a known name or ID	Text fields: (LO, SH, ST, LT, UT) searched, as whole words, for the names and IDs found elsewhere in the same file.
Burned-in text:	Names printed in the top-left corner of the images.

Cleaning (usable files only):
* PatientID and PatientName are replaced with a consistent pseudonym (PATIENT_001, ...) so a patient's videos stay linked.
* Other flagged fields are blanked rather than deleted, so the files remain valid DICOM.
* All private tags are removed.
* The burned-in text region is blacked out on every frame of every usable file, using a few boxes sized to the text lines so they stay clear of the ultrasound sector.
* PatientIdentityRemoved = YES and DeidentificationMethod are set.

Verification:
Each cleaned file is reopened and every header value is searched for the names and IDs found in the original. Any match, or any remaining private tag, is flagged. Burned-in removal is checked visually with the before/after previews.

Results:
Files in release: [160]
Usable videos: [134/134]
Distinct patients among usable videos: [105]
Cleaned files passing verification: [134/134]
