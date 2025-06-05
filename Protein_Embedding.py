import argparse
import torch
from tqdm import tqdm
import csv
from Bio import SeqIO
import esm
from transformers import T5Tokenizer, T5EncoderModel, BertModel, BertTokenizer


def esm2_embedding(input_fasta, output_csv):
    print(" Using ESM-2 model...")
    model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()

    with open(output_csv, mode='w', newline='') as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["ID"] + [f"emb_{i}" for i in range(2560)])

        for record in tqdm(SeqIO.parse(input_fasta, "fasta"), desc="Processing sequences"):
            seq_id = record.id
            sequence = str(record.seq)

            try:
                batch_labels, batch_strs, batch_tokens = batch_converter([(seq_id, sequence)])
                with torch.no_grad():
                    results = model(batch_tokens, repr_layers=[36], return_contacts=False)
                    token_representations = results["representations"][36]
                    embedding = token_representations[0, 1:len(sequence)+1].mean(0).cpu().tolist()
                writer.writerow([seq_id] + embedding)
            except Exception as e:
                print(f" Error processing {seq_id}: {e}")

def prot_t5_embedding(input_fasta, output_csv):
    print(" Using ProtT5 model...")
    tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False, legacy=False)
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
    model = model.eval()

    with open(output_csv, mode='w', newline='') as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["ID"] + [f"emb_{i}" for i in range(1024)])

        for record in tqdm(SeqIO.parse(input_fasta, "fasta"), desc="Processing sequences"):
            seq_id = record.id
            sequence = str(record.seq)
            sequence_spaced = ' '.join(list(sequence))  # Add spaces between residues

            try:
                ids = tokenizer(sequence_spaced, return_tensors="pt", padding=True)
                with torch.no_grad():
                    output = model(**ids)
                    embedding = output.last_hidden_state.mean(dim=1).squeeze().cpu().tolist()                    
                writer.writerow([seq_id] + embedding)
            except Exception as e:
                print(f" Error processing {seq_id}: {e}")

def prot_bert_embedding(input_fasta, output_csv):
    from transformers import BertTokenizer, BertModel
    print(" Using ProtBERT model...")
    tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
    model = BertModel.from_pretrained("Rostlab/prot_bert")
    model = model.eval()

    with open(output_csv, mode='w', newline='') as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["ID"] + [f"emb_{i}" for i in range(1024)])

        for record in tqdm(SeqIO.parse(input_fasta, "fasta"), desc="Processing sequences"):
            seq_id = record.id
            sequence = str(record.seq)
            sequence_spaced = ' '.join(list(sequence))  # Add spaces between amino acids

            try:
                inputs = tokenizer(sequence_spaced, return_tensors="pt")
                with torch.no_grad():
                    outputs = model(**inputs)
                    hidden_states = outputs.last_hidden_state  # shape: [1, seq_len, 1024]
                    embedding = hidden_states[0][1:-1].mean(dim=0).cpu().tolist()  # exclude [CLS] and [SEP]
                writer.writerow([seq_id] + embedding)
            except Exception as e:
                print(f" Error processing {seq_id}: {e}")
                
def main():
    parser = argparse.ArgumentParser(description="Protein sequence embedding generator.")
    parser.add_argument("-i", "--input", required=True, help="Input FASTA file")
    parser.add_argument("-o", "--output", required=True, help="Output CSV file")
    parser.add_argument("-m", "--method", required=True, choices=["esm2", "prot_t5", "prot_bert"], default="esm2", help="Embedding method")
    args = parser.parse_args()

    if args.method == "esm2":
        esm2_embedding(args.input, args.output)
    elif args.method == "prot_t5":
        prot_t5_embedding(args.input, args.output)
    elif args.method == "prot_bert":
        prot_bert_embedding(args.input, args.output)
        
if __name__ == "__main__":
    main()


"""
Protein Sequence Embedding Generator

This script generates protein embeddings using pre-trained models:
- ESM-2
- ProtT5
- ProtBERT

Usage:
    python Protein_Embedding.py -i input.fasta -o output.csv -m esm2
"""