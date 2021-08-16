# Abstractified Multi-Instance Learning (AMIL) for Biomedical Relation Extraction

Model parameters and settings can be tweaked via the `config.py` file in the project's root. All scripts and models were built using Python 3.8.5.

### **Step 1 - Obtain the Data:** 
- Download the [2019 PubMed corpus](https://mbr.nlm.nih.gov/Download/Baselines/2019/). For this, you can use the [Pubmed Downloader](https://github.com/elangovana/pubmed-downloader). Once downloaded, combine the files into a single text file `medline_abs.txt` and save it to the following directory: `/data/MEDLINE/raw/`
- Download the [2019AB UMLS Release Files](https://uts.nlm.nih.gov/uts/login?service=https:%2F%2Fdownload.nlm.nih.gov%2Fumls%2Fkss%2F2019AB%2Fumls-2019AB-full.zip). Once installed and extracted, move the files `MRCONSO.RRF`,`MRREL.RRF`,`MRSTY.RRF` to the directory `/data/UMLS/raw`.
  
### **Step 2 - Preprocessing:** 
- Install the project requirements with `pip install -r requirements.txt`.
- Run all the scripts in the preprocessing folder in order of the number in their file name (e.g. start with 1, end with 7).

### **Step 3 - Training:** 
- To train AMIL, run the `train.py` scipt in the project's root directory.

### **Step 4 - Evaluation:** 
- To evaluate AMIL, run the `eval.py` scipt in the project's root directory.

## Acknowledgements 
Thanks to Saadullah Amin for their assistance and providing the inspiration for this work. This repository is a fork of https://github.com/suamin/umls-medline-distant-re.
