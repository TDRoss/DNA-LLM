import random
import json
import numpy as np
from tqdm import tqdm
import nupack as nup

def reverse_complement(dna):
    """Return the reverse complement of a DNA sequence."""
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    return ''.join(complement[base] for base in reversed(dna))

def analyze_strands(strand1, strand2, nupackmodel):
    A = nup.Strand(strand1, name='A')
    B = nup.Strand(strand2, name='B')
    c1 = nup.Complex([A,B]) 
    complex_set = nup.ComplexSet(strands={A: 1e-8, B: 1e-8}, complexes=nup.SetSpec(max_size=0, include=[c1]))
    complex_analysis = nup.complex_analysis(complex_set, compute=['mfe','pfunc', 'pairs'], model=nupackmodel)
    complex_vals = complex_analysis[c1]
    return complex_vals


def generate_secondary_structure(seq_length,num_mismatches):
    dp_comp = {'(': ')', '.':'.'}
    start_strand = list('('*seq_length)
    allinds = list(range(0,seq_length))
    mis_inds = random.sample(allinds,num_mismatches)
    for ind in mis_inds:
        start_strand[ind] = '.'
    end_strand =  [dp_comp[base] for base in reversed(start_strand)]
    dotpar = ''.join(start_strand + list('+') + end_strand)
    return dotpar

def sequence_design(dotpar,seq_length,nupackmodel):
    f = nup.Domain(f'N{seq_length}', name='f')
    g = nup.Domain(f'N{seq_length}', name='g')
    F = nup.TargetStrand([f], name='Strand F')
    G = nup.TargetStrand([g], name='Strand G')
    Ct = nup.TargetComplex([F,G], dotpar, name='Ct')
    t1 = nup.TargetTube(on_targets={Ct: 1e-8}, name='t1')
    # sim1 = nup.Similarity([f,g], f'S{seq_length*2}', limits=[0.3, 0.7])
    my_design = nup.tube_design(tubes=[t1], hard_constraints=[], soft_constraints=[], model=nupackmodel)
    while True:
        try:
            my_results = my_design.run(trials=1)
            strand1 = str(my_results[0].to_analysis(F))
            strand2 = str(my_results[0].to_analysis(G))
            break
        except:
            None
    return strand1, strand2


def get_sequence(seq_length,num_mismatches, nupackmodel):
    dotpar = generate_secondary_structure(seq_length,num_mismatches)
    strand1, strand2 = sequence_design(dotpar,seq_length,nupackmodel)
    return strand1,strand2

def generate_training_structures():
    training_size = 11000
    nupackmodel = nup.Model(material='DNA',celsius=20)
    all_data = []
    structures = set()
    with tqdm(total=training_size) as pbar: 
        while len(structures) < training_size:
            seq_len = random.randint(10,25)
            num_mismatches = max(1,random.randint(0,round(seq_len*0.3)))
            # while True: #keep generating structures until a unique one is found
            dotpar = generate_secondary_structure(seq_len,num_mismatches)
            if dotpar not in structures:
                strand1, strand2 = sequence_design(dotpar,seq_len,nupackmodel)
                complex_vals = analyze_strands(strand1,strand2,nupackmodel)
                mfe_dotpar = str(complex_vals.mfe[0].structure)
                if mfe_dotpar == dotpar:
                    structures.add((dotpar))
                    all_data.append((dotpar,strand1,strand2))
                    pbar.update(1)
                              

    # Randomly generate indices for train and validation split
    train_indices = random.sample(range(training_size), training_size-1000)
    val_indices = list(set(range(training_size)) - set(train_indices))

    # Split using the generated indices
    train_set = [all_data[i] for i in train_indices]
    val_set = [all_data[i] for i in val_indices]

    with open(f"training_data/structure_train_set.json", 'w') as f:
        json.dump(train_set,f)

    with open(f"training_data/structure_validation_set.json", 'w') as f:
        json.dump(val_set,f)

if __name__ == '__main__':
    random.seed(23)
    generate_training_structures()

