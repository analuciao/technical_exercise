1. What problems did you find in the DICOM files, and what should be fixed before they are shared?

Of 160 files in the release, 134 are usable videos from 105 distinct patients. The rest had problems:

Unusable files: 2 could not be opened as DICOM, 3 failed the SHA256 fingerprint check (likely corrupted or altered), 6 were duplicates of other files, 7 were single images rather than videos, and 5 were shorter than one heartbeat (under 1 second at 60 bpm). 2 files listed in echo-record-list.csv were missing from the folder. These should be removed or replaced, and the missing files located.

Patient information the earlier anonymization missed: [e.g. real MRNs in PatientID, names in vendor-specific private tags, and unshifted dates]. Dates in this release were shifted to around 2150, so any date in 2026 or earlier was treated as real. Some videos also had the patient’s name printed in the top-left corner of the image.

Before sharing, the data should be replaced with the cleaned copies my script produces. These replace names and IDs with consistent pseudonyms (so each patient’s videos stay linked), blank other identifying fields, remove private tags, and black out the printed text on every frame. Every cleaned file was reopened and checked that none of the names or IDs found remain (134/134 passed). 

1. How did your measurements compare with the experts’? Where did your method struggle, and why?

I fine-tuned a U-Net (ImageNet-pretrained resnet34 encoder) on about 15,000 expert-traced frames from the training split. Training a model fit better than manual drawing or classic image processing, given the amount of labeled data and how noisy echo images are. On held-out data, predicted outlines overlapped the experts’ with a mean Dice score of 0.909. Fractional area change (FAC), a 2D proxy for ejection fraction, differed from the experts’ by 5.4 points on average, and correlated with the dataset’s EF at 0.75.

3. One thing you would do differently with more time

I would add data augmentation during training: small rotations and brightness/contrast changes. Echo images vary considerably in probe angle and gain between patients and machines, and augmentation would expose the model to that variety. That should reduce the cases where it struggled on faint or unusually positioned ventricles.