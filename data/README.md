# Data

This project uses the **MeQSum** corpus — 1,000 consumer health questions (`CHQ`), each paired with a short
expert-written summary (`Summary`).

The dataset is **not stored in this repository**. The notebook downloads it automatically into this folder from the
official repository: <https://github.com/abachaa/MeQSum>

| Column | Description |
|---|---|
| `File` | Source file identifier |
| `CHQ` | Original consumer health question (often long and noisy) |
| `Summary` | Reference summary of the question |

MeQSum is released under the [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license. If you use it, cite:

> Asma Ben Abacha and Dina Demner-Fushman. *On the Summarization of Consumer Health Questions.* ACL 2019.
> <https://aclanthology.org/P19-1215>
