import json
import time
import multiprocessing

from Bio import Align
import re
from openai import OpenAI
client = OpenAI()

def reverse_complement(dna):
    """Return the reverse complement of a DNA sequence."""
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    return ''.join(complement[base] for base in reversed(dna))

def generate_secondary_structure_jsonl(seqs, output_filename):
    with open(output_filename, 'w') as f:
        for seq1, seq2, _, prob_string, dotpar in seqs:
            # stepbystep = []
            # basecomp = []
            # rev2 = reverse_complement(seq2)
            # for char1, char2 in zip(seq1, rev2):
            #     bit = int(char1==char2)
            #     stepbystep.append(f"{char1}{char2}:{bit}")
            #     basecomp.append(str(bit))    
            # step_string = ''.join(stepbystep).strip()
            # base_compare_string = ''.join(basecomp).strip()
            # message = {
            #     "messages": [
            #         {"role": "system", "content": "Please return the dot-parens-plus secondary structure for the following DNA sequence pair and pairing binary string."},
            #         {"role": "user", "content": f"{seq1} {rev2} {base_compare_string}"},
            #         {"role": "assistant", "content": f"{dotpar}"},
            #     ]
            # }
            stepbystep = []
            rev2 = reverse_complement(seq2)
            base_compare_string = ''.join(['1' if rev2[i] == seq1[i] else '0' for i in range(len(rev2))])
            pad = '__'
            pad_seq1 = pad+seq1+pad
            pad_rev2 = pad+rev2+pad
            pad_bc = pad+base_compare_string+pad
            pad_dp1 = 5*'x'+dotpar[:len(seq1)]
            pad_dp2 = 5*'x'+dotpar[-len(seq1):][::-1]
            for baseind in range(len(seq1)):
                indx = slice(baseind,baseind+5)
                stepbystep.append(f"[{pad_seq1[indx]},{pad_rev2[indx]},{pad_bc[indx]},{pad_dp1[indx]},{pad_dp2[indx]}]:{dotpar[baseind]},{dotpar[-len(seq1):][::-1][baseind]} ")    
            # for char1, char2 in zip(seq1, rev2):
            #     bit = int(char1==char2)
            #     stepbystep.append(f"{char1}{char2}:{bit} ")
            step_string = ''.join(stepbystep).strip()
            message = {
                "messages": [
                    {"role": "system", "content": "You are a DNA analyzer. Please compare the two sequences and the corresponding base matching binary to generate sections of the secondary structure in paren-dot-plus notation."},
                    {"role": "user", "content": f"{seq1} {rev2} {base_compare_string}"},
                    {"role": "assistant", "content": f"{step_string} ans:{dotpar[:len(seq1)]} {dotpar[-len(seq1):][::-1]}"},
                ]
            }
            f.write(json.dumps(message) + '\n')

def generate_align_jsonl(seqs, output_filename):
    with open(output_filename, 'w') as f:
        for seq1, seq2, _, prob_string, dotpar in seqs:
            stepbystep = []
            rev2 = reverse_complement(seq2)
            base_compare_string = ''.join(['1' if rev2[i] == seq1[i] else '0' for i in range(len(rev2))])
            # aligner = Align.PairwiseAligner()
            # alignment = aligner.align(seq1,rev2)
            # align_string = str(alignment[0])
            # pattern = r'\d\s+([A-Z-]+)\s+\d+'
            # matches = re.findall(pattern, align_string)
            pad_seq1 = '__'+seq1+'__'
            pad_rev2 = '__'+rev2+'__'
            for baseind in range(len(seq1)):
                # if baseind == 0:
                #     indx = slice(0,2)
                # elif baseind == len(seq1)-1:
                #     indx = slice(baseind-1,baseind+1)    
                # else:
                indx = slice(baseind,baseind+5)
                stepbystep.append(f"({pad_seq1[indx]},{pad_rev2[indx]}):({prob_string[baseind]},{prob_string[-len(seq1):][::-1][baseind]}) ")    
            # for char1, char2 in zip(seq1, rev2):
            #     bit = int(char1==char2)
            #     stepbystep.append(f"{char1}{char2}:{bit} ")
            step_string = ''.join(stepbystep).strip()
            message = {
                "messages": [
                    {"role": "system", "content": "You are a DNA analyzer. Please compare the two sequences and the corresponding base comparison binary to provide two binary strings indicating which bases bind"},
                    {"role": "user", "content": f"{seq1} {rev2} {base_compare_string}"},
                    {"role": "assistant", "content": f"{step_string} ans:{prob_string[:len(seq1)]} {prob_string[-len(seq1):][::-1]}"},
                ]
            }
            f.write(json.dumps(message) + '\n') 


def generate_base_probability_jsonl(seqs, output_filename):
    with open(output_filename, 'w') as f:
        for seq1, seq2, _, prob_string, dotpar in seqs:
            stepbystep = []
            rev2 = reverse_complement(seq2)
            for baseind in range(len(seq1)):
                if baseind == 0:
                    indx = slice(0,2)
                elif baseind == len(seq1)-1:
                    indx = slice(baseind-1,baseind+1)    
                else:
                    indx = slice(baseind-1,baseind+2)
                stepbystep.append(f"({seq1[indx]},{rev2[indx]}):({prob_string[baseind]},{prob_string[-len(seq1):][::-1][baseind]}) ")    
            # for char1, char2 in zip(seq1, rev2):
            #     bit = int(char1==char2)
            #     stepbystep.append(f"{char1}{char2}:{bit} ")
            step_string = ''.join(stepbystep).strip()
            message = {
                "messages": [
                    {"role": "system", "content": "Please return a binary string to indicate which bases match for the following DNA sequence pair."},
                    {"role": "user", "content": f"{seq1} {rev2}"},
                    {"role": "assistant", "content": f"{step_string} ans:{prob_string[:len(seq1)]} {prob_string[-len(seq2):][::-1]}"},
                ]
            }
            f.write(json.dumps(message) + '\n')            


def run_fine_tune_job(args):
    experiment, train_size, testnum = args
    #Load training file
    training_file = client.files.create(
    file=open(f"fine_tune_sets/{experiment}_test_{testnum}_train_size_{train_size}.jsonl", "rb"),
    purpose='fine-tune'
    )
    print("Uploaded file id", training_file.id)

    while True:
        file_handle = client.files.retrieve(training_file.id)
        print(f"Fine-tuning status: {file_handle.status}")
        if file_handle.status == "processed":
            print("File processed")
            break
        time.sleep(10)

    #Start fine-tuning
    ftjob = client.fine_tuning.jobs.create(
    training_file=training_file.id,
    model="gpt-3.5-turbo-1106"
    )
    print(ftjob.id)
    while True:
        job_handle = client.fine_tuning.jobs.retrieve(ftjob.id)
        print(f"Fine-tuning status: {job_handle.status}")
        if job_handle.status == "succeeded":
            print("Fine-tuning complete")
            print("Fine-tuned model info", job_handle)
            print("Model id", job_handle.fine_tuned_model)
            break
        time.sleep(60) 
    return job_handle.fine_tuned_model    


def fine_tune(experiment,train_size,condition=None):
    with open(f"training_data/DNA_sequence_train_set.json", 'r') as f: 
            train_set = json.load(f)
    for ts in train_sizes:
        if experiment == "secondary_structure":
            generate_secondary_structure_jsonl(train_set[:ts],f"fine_tune_sets/{experiment}_test_{testnum}_train_size_{ts}.jsonl")
        elif experiment == "base_probability":
            generate_base_probability_jsonl(train_set[:ts],f"fine_tune_sets/{experiment}_train_size_{ts}.jsonl")
        # generate_prob_jsonl(train_set[0:train_size],f"fine_tune_sets/{experiment}_train_size_{train_size}.jsonl")
        # generate_struc_jsonl(train_set[0:train_size],f"fine_tune_sets/{experiment}_train_size_{train_size}.jsonl")

    arguments = [(experiment, ts,testnum) for ts in train_sizes]
    with multiprocessing.Pool(3) as pool:
        model_ids = pool.map(run_fine_tune_job, arguments)

    model_list = list(zip(train_sizes,model_ids))
    with open(f"model_ids/{experiment}_test_{testnum}_models.json", 'w') as f:
        json.dump(model_list, f)



if __name__ == '__main__':
    experiment = "base_comparison"
    testnum = 1
    # train_sizes = [200, 500, 1400, 3700, 10000]
    train_sizes = [10000]
    fine_tune(experiment,train_sizes)

    

    