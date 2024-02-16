import json

def reverse_complement(dna):
    """Return the reverse complement of a DNA sequence."""
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    return ''.join(complement[base] for base in reversed(dna))

with open(f"training_data/sequence_train_set.json", 'r') as f: 
        train_set = json.load(f)

data = []
for seq1, seq2, _, prob_string, dotpar in train_set:
    rev2 = reverse_complement(seq2)
    base_compare_string = ''.join(['1' if rev2[i] == seq1[i] else '0' for i in range(len(rev2))])
    stepbystep = []
    for char1, char2, bit in zip(seq1,seq2[::-1], base_compare_string):
        stepbystep.append(f"{char1}{char2}:{bit} ")
    step_string = ''.join(stepbystep).strip()

    # tdic = {"sequences":f"{seq1} {seq2}","base_compare": f"{base_compare_string}"}
    tdic = {"sequences":f"{seq1} {seq2}","base_compare": f"{step_string} ans:{base_compare_string}"}
    data.append(tdic)

with open("training_data/dna_base_compare.json", 'w') as file:
    for entry in data:
        json_line = json.dumps(entry)
        file.write(json_line + '\n')