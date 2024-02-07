import random
import json
import numpy as np
from tqdm import tqdm
import nupack as nup

def reverse_complement(dna):
    """Return the reverse complement of a DNA sequence."""
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    return ''.join(complement[base] for base in reversed(dna))

def generate_sequence(seq_len,num_mismatches):
    nucleotides = 'ATCG'
    seq1 = ''.join(random.choice(nucleotides) for _ in range(seq_len))
    rev_comp = list(reverse_complement(seq1))
    #Introduce mismatches
    mismatch_index = random.sample(range(0,seq_len), num_mismatches)
    for indx in mismatch_index:
        rev_comp[indx] = random.choice([n for n in nucleotides if n != rev_comp[indx]])
    seq2 = ''.join(rev_comp)
    return seq1, seq2


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
    sim1 = nup.Similarity([f,g], f'S{seq_length*2}', limits=[0.3, 0.7])
    my_design = nup.tube_design(tubes=[t1], hard_constraints=[sim1], soft_constraints=[], defect_weights=None, options=None, model=nupackmodel)
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
    # bad_probs = 0    
    # while probability < min_prob: #Generate new sequences until equilibrium probability is met
        # if bad_probs == 3:
        #     dotpar = generate_secondary_structure(seq_length,num_mismatches)
        #     bad_probs = 0
    strand1, strand2 = sequence_design(dotpar,seq_length,nupackmodel)
    return strand1,strand2

def generate_training_sequences():
    training_size = 11000
    nupackmodel = nup.Model(material='DNA',celsius=20)
    all_data = []
    seqs = set()
    with tqdm(total=training_size) as pbar: 
        while len(seqs) < training_size:
            seq_len = random.randint(10,25)
            num_mismatches = max(1,random.randint(0,round(seq_len*0.3)))
            while True: #keep generating sequence pairs until a unique set is found
                # seq1, seq2 = get_sequence(seq_len,num_mismatches,nupackmodel)
                seq1, seq2 = generate_sequence(seq_len,num_mismatches)
                if (seq1,seq2) not in seqs and (seq2, seq1) not in seqs:
                    complex_vals = analyze_strands(seq1,seq2,nupackmodel)
                    mfe = round(complex_vals.mfe[0].energy,1)
                    dotpar = str(complex_vals.mfe[0].structure)
                    #Get base-pair probabilities
                    pair_array = complex_vals.pairs.to_array()
                    np.fill_diagonal(pair_array, 0)
                    rounded_array=np.around(pair_array)
                    total_pair_prob = rounded_array.sum(axis=0)
                    prob_string = np.array2string(total_pair_prob, separator='', max_line_width=np.inf).replace('[', '').replace(']', '').replace(',', '').replace('.', '')
                    prob_struc = True
                    splitdotpar = dotpar.split('+')#Ensure that there is no self-complementarity
                    if ')' in splitdotpar[0] or '(' in splitdotpar[1]:
                        prob_struc = False
                    if prob_struc:
                        for char1, char2 in zip(prob_string, dotpar.replace('+','')): #Ensure rounded probabilities match the secondary structure
                            if char1 == '0' and char2 != '.':
                                prob_struc = False
                                break
                            elif char1 == '1' and char2 not in '()':
                                prob_struc = False
                                break
                        if prob_struc:
                            all_data.append((seq1,seq2,mfe,prob_string,dotpar))
                            seqs.add((seq1,seq2))
                            pbar.update(1)
                            break                       

#                     


# def generate_training_sequences(training_size):
#     nupackmodel = nup.Model(material='DNA',celsius=20)
#     all_data = []
#     seqs = set()
#     with tqdm(total=training_size) as pbar: 
#         while len(seqs) < training_size:
#             seq_len = random.randint(10,25)
#             num_mismatches = max(1,random.randint(0,round(seq_len*0.3)))
#             while True:
#                 seq1, seq2 = generate_sequence(seq_len,num_mismatches)
#                 if (seq1,seq2) not in seqs and (seq2,seq1) not in seqs: #Ensure sequence pair is unique
#                     complex_vals = analyze_strands(seq1,seq2,nupackmodel)
#                     mfe = round(complex_vals.mfe[0].energy,1)
#                     dotpar = str(complex_vals.mfe[0].structure)
#                     #Get base-pair probabilities
#                     pair_array = complex_vals.pairs.to_array()
#                     np.fill_diagonal(pair_array, 0)
#                     rounded_array=np.around(pair_array)
#                     total_pair_prob = rounded_array.sum(axis=0)
#                     prob_string = np.array2string(total_pair_prob, separator='', max_line_width=np.inf).replace('[', '').replace(']', '').replace(',', '').replace('.', '')
#                     prob_struc = True
#                     splitdotpar = dotpar.split('+')#Ensure that there is no self-complementarity
#                     if ')' in splitdotpar[0] or '(' in splitdotpar[1]:
#                         prob_struc = False
#                     if prob_struc:
#                         for char1, char2 in zip(prob_string, dotpar.replace('+','')): #Ensure rounded probabilities match the secondary structure
#                             if char1 == '0' and char2 != '.':
#                                 prob_struc = False
#                                 break
#                             elif char1 == '1' and char2 not in '()':
#                                 prob_struc = False
#                                 break
#                         if prob_struc:
#                             all_data.append((seq1,seq2,mfe,prob_string,dotpar))
#                             seqs.add((seq1,seq2))
#                             pbar.update(1)
#                             break

    # Randomly generate indices for train and validation split
    train_indices = random.sample(range(training_size), training_size-1000)
    val_indices = list(set(range(training_size)) - set(train_indices))

    # Split using the generated indices
    train_set = [all_data[i] for i in train_indices]
    val_set = [all_data[i] for i in val_indices]

    with open(f"training_data/alignment_train_set.json", 'w') as f:
        json.dump(train_set,f)

    with open(f"training_data/alignment_validation_set.json", 'w') as f:
        json.dump(val_set,f)

if __name__ == '__main__':
    random.seed(23)
    generate_training_sequences()

