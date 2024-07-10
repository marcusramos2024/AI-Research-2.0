import pandas as pd
from esm import FastaBatchedDataset, pretrained
import pathlib
import torch

def csv_fasta(file_path):
    df = pd.read_csv(file_path)
    with open(file_path.replace(".csv", ".fasta"), 'a') as file:        
        for index, row in df.iterrows():
            file.write(">" + row["Protein"] + "\n" + row["sequence"] + "\n")

def ESM_extract_embeddings(model_name, model_layers, fasta_file, tokens_per_batch=4096, seq_length=1022):
    repr_layers = []
    
    for i in range(1,model_layers + 1):
        repr_layers.append(i)

    
    model, alphabet = pretrained.load_model_and_alphabet(model_name)
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()
        
    dataset = FastaBatchedDataset.from_file(fasta_file)
    batches = dataset.get_batch_indices(tokens_per_batch, extra_toks_per_seq=1)

    data_loader = torch.utils.data.DataLoader(
        dataset, 
        collate_fn=alphabet.get_batch_converter(seq_length), 
        batch_sampler=batches
    )

    output_dir = pathlib.Path(model_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            with open("LOG.txt", "w") as file:
                file.write(f'Processing batch {batch_idx + 1} of {len(batches)}\n')

            if torch.cuda.is_available():
                toks = toks.to(device="cuda", non_blocking=True)

            out = model(toks, repr_layers=repr_layers, return_contacts=False)

            logits = out["logits"].to(device="cpu")
            representations = {layer: t.to(device="cpu") for layer, t in out["representations"].items()}
            
            for i, label in enumerate(labels):
                entry_id = label.split()[0]
                
                filename = output_dir / f"{entry_id}.pt"
                truncate_len = min(seq_length, len(strs[i]))

                result = {"entry_id": entry_id}
                result["mean_representations"] = {
                        layer: t[i, 1 : truncate_len + 1].mean(0).clone()
                        for layer, t in representations.items()
                    }

                torch.save(result, filename)



# models = {
#     "esm2_t6_8M_UR50D" : 6,
#     "esm2_t12_35M_UR50D" : 12,
#     "esm2_t30_150M_UR50D" : 30,
#     "esm2_t33_650M_UR50D" : 33,
#     "esm2_t36_3B_UR50D" : 36,
#     "esm2_t48_15B_UR50D": 48,
# }

# for model in models:
#     ESM_extract_embeddings(model, models[model], "Base.fasta")


ESM_extract_embeddings("esm2_t33_650M_UR50D", 33, "Base.fasta")
