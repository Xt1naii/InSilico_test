# InSilico_test
Research report for the job assessment test for the biomedical data scientist position

The main purpose of the test is to obtain a pipeline for collecting drug information, processing
and creating a binary classification predictive model.

Quick start
```
python main.py
```

files in data/
all_proteins_kegg.parquet -- get_kegg_for_string.py output
c2.cp.kegg.v2023.1.Hs.entrez.gmt.txt -- KEGG pathways
mechanisms_ensg.parquet -- get_ensg.py output
scrape_result.parquet -- get_adinsight_data.py

files in artifacts/
model_dict.pt -- model.py output, the trained model’s learned parameters
node_embeddings.pkl -- model.py output
target_dict.pkl -- get_target.py output, targets for all proteins from STRING PPI network
vector_dict.pkl -- get_kegg_dict.py output, dict with pathway vectors for all proteins from STRING PPI network
