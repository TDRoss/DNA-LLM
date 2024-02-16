import json
import time
import multiprocessing
from openai import OpenAI
client = OpenAI()

def reverse_complement(dna):
    """Return the reverse complement of a DNA sequence."""
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    return ''.join(complement[base] for base in reversed(dna))

def generate_reverse_complement_jsonl(condition,seqs, output_filename):
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    with open(output_filename, 'w') as f:
        for seq1, seq2, _, prob_string, dotpar in seqs:
            rev2 = reverse_complement(seq2)
            system_message = {"role": "system", "content": "You are a DNA analyzer. Please return the reverse complement of the following sequence."}
            user_message = {"role": "user", "content": f"{seq2}"}
            if condition == "naive":
                assistant_message = {"role": "assistant", "content": f"{rev2}"}
            elif condition == "CoT":
                stepbystep=[]
                for indx in range(len(rev2)):
                    stepbystep.append(f"{seq2[:len(seq2)-indx]},{seq2[-(indx+1)]}:{rev2[:indx+1]} ")
                step_string = ''.join(stepbystep).strip()
                assistant_message = {"role": "assistant", "content": f"{step_string} ans:{rev2}"}              
            message = {"messages": [system_message,user_message,assistant_message]}
            f.write(json.dumps(message) + '\n')

def generate_base_comparison_jsonl(condition,seqs, output_filename):
    with open(output_filename, 'w') as f:
        for seq1, seq2, _, prob_string, dotpar in seqs:
            rev2 = reverse_complement(seq2)
            base_compare_string = ''.join(['1' if rev2[i] == seq1[i] else '0' for i in range(len(rev2))])
            if condition == "-rev_comp":
                system_message = {"role": "system", "content": "You are a DNA analyzer. Please compare the two partially complementary sequences and return a binary string corresponding to a valid or invalid base pairing."}
                user_message = {"role": "user", "content": f"{seq1} {seq2}"}
                assistant_message = {"role": "assistant", "content": f"{base_compare_string}"}
            elif condition == "+rev_comp":
                system_message = {"role": "system", "content": "You are a DNA analyzer. Please compare the two sequences and return a binary string corresponding to characters being identical or not."}
                user_message = {"role": "user", "content": f"{seq1} {rev2}"}
                assistant_message = {"role": "assistant", "content": f"{base_compare_string}"}
            elif condition == "CoT":
                complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
                stepbystep=[]
                for char1, char2 in zip(seq1, seq2[::-1]):
                    bit = int(complement[char1]==char2)
                    stepbystep.append(f"{char1}{char2}:{bit} ")
                step_string = ''.join(stepbystep).strip()
                system_message = {"role": "system", "content": "You are a DNA analyzer. Please compare the two partially complementary sequences and return a binary string corresponding to a valid or invalid base pairing."}
                user_message = {"role": "user", "content": f"{seq1} {seq2}"}
                assistant_message = {"role": "assistant", "content": f"{step_string} ans:{base_compare_string}"}
            elif condition == "revCoT":
                stepbystep=[]
                for indx, (char1, char2, bit) in enumerate(zip(seq1, rev2, base_compare_string)):
                    stepbystep.append(f"{char1}{char2}:{bit} ")
                    # stepbystep.append(f"({seq1[indx:]},{rev2[indx:]}){char1}{char2}:{bit} ")
                step_string = ''.join(stepbystep).strip()
                system_message = {"role": "system", "content": "You are a DNA analyzer. Please compare the two partially complementary sequences and return a binary string corresponding to a valid or invalid base pairing."}
                user_message = {"role": "user", "content": f"{seq1} {seq2}"}
                assistant_message = {"role": "assistant", "content": f"{rev2} {step_string} ans:{base_compare_string}"}                
            elif condition == "CoT+rev_comp":
                stepbystep=[]  
                for indx, (char1, char2, bit) in enumerate(zip(seq1, rev2, base_compare_string)):
                    # stepbystep.append(f"{char1}{char2}:{bit} ")
                    stepbystep.append(f"({seq1[indx:]},{rev2[indx:]}){char1}{char2}:{bit} ")
                step_string = ''.join(stepbystep).strip()
                system_message = {"role": "system", "content": "You are a DNA analyzer. Please compare the two sequences and return a binary string corresponding to characters being identical or not."}
                user_message = {"role": "user", "content": f"{seq1} {rev2}"}
                assistant_message = {"role": "assistant", "content": f"{step_string} ans:{base_compare_string}"}
            elif condition == "CoT+rev_comp_to_dotpar":
                stepbystep=[]  
                for indx, (char1, char2) in enumerate(zip(seq1, rev2)):
                    # stepbystep.append(f"{char1}{char2}:{bit} ")
                    stepbystep.append(f"({seq1[indx:]},{rev2[indx:]}){char1}{char2}:{dotpar[indx]} ")
                step_string = ''.join(stepbystep).strip()
                system_message = {"role": "system", "content": "You are a DNA analyzer. Please compare the two sequences and return the secondary structure using dot-parens-plus notation."}
                user_message = {"role": "user", "content": f"{seq1} {rev2}"}
                assistant_message = {"role": "assistant", "content": f"{step_string} ans:{dotpar}"}                

            message = {"messages": [system_message,user_message,assistant_message]}
            f.write(json.dumps(message) + '\n')

def generate_base_pairing_jsonl(condition,seqs, output_filename):
    with open(output_filename, 'w') as f:
        for seq1, seq2, _, prob_string, dotpar in seqs:
            rev2 = reverse_complement(seq2)
            base_compare_string = ''.join(['1' if rev2[i] == seq1[i] else '0' for i in range(len(rev2))]) 
            pad = '__'
            zpad = "00"
            pad_seq1 = pad+seq1+pad
            pad_rev2 = pad+rev2+pad
            pad_bc = pad+base_compare_string+pad
            zpad_bc = zpad+base_compare_string+zpad
            pad_bp1 = 5*'x'+prob_string[:len(seq1)]
            pad_bp2 = 5*'x'+prob_string[-len(seq1):][::-1]
            stepbystep =[]
            for baseind in range(len(seq1)):
                indx = slice(baseind,baseind+5)
                if condition == "+base_comparison" or condition == "2altdotparens+base_comparison":
                    stepbystep.append(f"[{pad_seq1[indx]},{pad_rev2[indx]},{pad_bc[indx]},{pad_bp1[indx]},{pad_bp2[indx]}]:{prob_string[baseind]},{prob_string[-len(seq1):][::-1][baseind]} ")
                elif condition == "3altdotparens+base_comparison" or "alt+base_comparison":
                    stepbystep.append(f"[{pad_seq1[indx]},{pad_rev2[indx]},{zpad_bc[indx]}]:{prob_string[baseind]},{prob_string[-len(seq1):][::-1][baseind]} ")
                # elif condition == "alt+base_comparison":
                #     mod_seq1 = ""
                #     mod_rev2 = ""
                #     for s1,r2,zb in zip(pad_seq1,pad_rev2,zpad_bc):
                #         if zb == "1":
                #             mod_seq1 += s1
                #             mod_rev2 += r2
                #         else:
                #             mod_seq1 += "_"
                #             mod_rev2 += "_"
                #     stepbystep.append(f"[{zpad_bc[indx]},{mod_seq1[indx]},{mod_rev2[indx]},{pad_bp1[indx]},{pad_bp2[indx]}]:{prob_string[baseind]},{prob_string[-len(seq1):][::-1][baseind]} ")    
                elif condition == "-base_comparison":
                    stepbystep.append(f"[{pad_seq1[indx]},{pad_rev2[indx]},{pad_bp1[indx]},{pad_bp2[indx]}]:{prob_string[baseind]},{prob_string[-len(seq1):][::-1][baseind]} ")
                elif condition == "dotparens+base_comparison":
                    stepbystep.append(f"[{pad_seq1[indx]},{pad_rev2[indx]},{pad_bc[indx]},{pad_bp1[indx]},{pad_bp2[indx]}]:{dotpar[baseind]},{dotpar[-len(seq1):][::-1][baseind]} ")
            step_string = ''.join(stepbystep).strip()

            if condition == "+base_comparison":
                system_message = {"role": "system", "content": "You are a DNA analyzer. Please compare the two sequences and the corresponding base comparison binary to generate two binaries representing where base pairing occurs."}
                user_message = {"role": "user", "content": f"{seq1} {rev2} {base_compare_string}"}
                assistant_message = {"role": "assistant", "content": f"{step_string} ans:{prob_string[:len(seq1)]} {prob_string[-len(seq1):][::-1]}"}
            elif condition == "-base_comparison":
                system_message = {"role": "system", "content": "You are a DNA analyzer. Please compare the two sequences to generate two binaries representing where base pairing occurs."}
                user_message = {"role": "user", "content": f"{seq1} {rev2}"}
                assistant_message = {"role": "assistant", "content": f"{step_string} ans:{prob_string[:len(seq1)]} {prob_string[-len(seq1):][::-1]}"}
            elif condition == "dotparens+base_comparison":
                system_message = {"role": "system", "content": "You are a DNA analyzer. Please compare the two sequences and the corresponding base comparison binary to show where bases bind using parens-dot-plus notation."}
                user_message = {"role": "user", "content": f"{seq1} {rev2} {base_compare_string}"}
                assistant_message = {"role": "assistant", "content": f"{step_string} ans:{dotpar[:len(seq1)]} {dotpar[-len(seq1):][::-1]}"}  
            elif condition == "alt+base_comparison":
                system_message = {"role": "system", "content": "You are a DNA analyzer. Please compare the two sequences and the corresponding base comparison binary to generate two binaries representing where base pairing occurs."}
                user_message = {"role": "user", "content": f"{seq1} {rev2} {base_compare_string}"}
                assistant_message = {"role": "assistant", "content": f"{step_string} ans:{prob_string[:len(seq1)]} {prob_string[-len(seq1):][::-1]}"}
            elif condition == "3altdotparens+base_comparison":
                system_message = {"role": "system", "content": "You are a DNA analyzer. Please compare the two sequences and the corresponding base comparison binary to determine the secondary structure in dot-parens-plus notation."}
                user_message = {"role": "user", "content": f"{seq1} {rev2} {base_compare_string}"}
                assistant_message = {"role": "assistant", "content": f"{step_string} {prob_string[:len(seq1)]} {prob_string[-len(seq1):][::-1]} ans:{dotpar}"}
            elif condition == "2altdotparens+base_comparison":
                system_message = {"role": "system", "content": "You are a DNA analyzer. Please compare the two sequences and the corresponding base comparison binary to determine the secondary structure in dot-parens-plus notation."}
                user_message = {"role": "user", "content": f"{seq1} {rev2} {base_compare_string}"}
                assistant_message = {"role": "assistant", "content": f"{step_string} {prob_string[:len(seq1)]} {prob_string[-len(seq1):][::-1]} ans:{dotpar}"}
            # elif condition == "alt+base_comparison":
            #     mod_seq1 = ""
            #     mod_rev2 = ""
            #     pad = '__'
            #     zpad = "00"
            #     pad_seq1 = pad+seq1+pad
            #     pad_rev2 = pad+rev2+pad
            #     pad_bc = pad+base_compare_string+pad
            #     zpad_bc = zpad+base_compare_string+zpad
            #     for s1,r2,zb in zip(pad_seq1,pad_rev2,zpad_bc):
            #         if zb == "1":
            #             mod_seq1 += s1
            #             mod_rev2 += r2
            #         else:
            #             mod_seq1 += "_"
            #             mod_rev2 += "_"
            #     system_message = {"role": "system", "content": "You are a DNA analyzer. Please compare the two sequences and the corresponding base comparison binary to generate two binaries representing where base pairing occurs."}
            #     user_message = {"role": "user", "content": f"{seq1} {rev2} {base_compare_string}"}
            #     assistant_message = {"role": "assistant", "content": f"{mod_seq1} ans:{prob_string[:len(seq1)]} {prob_string[-len(seq1):][::-1]}"}                                       
            message = {"messages": [system_message,user_message,assistant_message]}
            f.write(json.dumps(message) + '\n')

def generate_convert_to_dotparens_jsonl(condition,seqs, output_filename):
    with open(output_filename, 'w') as f:
        for seq1, seq2, _, prob_string, dotpar in seqs:
            if condition == "complete":
                system_message = {"role": "system", "content": "You are a DNA analyzer. Please take the base pairing binaires and convert them to parens-dot-plus notation."}
                user_message = {"role": "user", "content": f"{prob_string[:len(seq1)]} {prob_string[-len(seq1):][::-1]}"}
                assistant_message = {"role": "assistant", "content": f"{dotpar}"}
            elif condition == "flip+complete":
                system_message = {"role": "system", "content": "You are a DNA analyzer. Please take the base pairing binaires and convert them to parens-dot-plus notation."}
                user_message = {"role": "user", "content": f"{prob_string[:len(seq1)]} {prob_string[-len(seq1):][::-1]}"}
                assistant_message = {"role": "assistant", "content": f"{prob_string[:len(seq1)]} {prob_string[-len(seq1):]} ans:{dotpar}"}
            elif condition == "char_convert":
                system_message = {"role": "system", "content": "You are a DNA analyzer. Please take the base pairing binaires and convert the characters to parens-dot notation."}
                user_message = {"role": "user", "content": f"{prob_string[:len(seq1)]} {prob_string[-len(seq1):][::-1]}"}
                assistant_message = {"role": "assistant", "content": f"{dotpar[:len(seq1)]} {dotpar[-len(seq1):][::-1]}"}  
            elif condition == "CoT":
                seq1prob = prob_string[:len(seq1)]
                rev2prob = prob_string[-len(seq2):][::-1]
                stepbystep=[]
                for indx, (char1, char2) in enumerate(zip(dotpar[:len(seq1)], dotpar[-len(seq2):][::-1])):
                    stepbystep.append(f"[{seq1prob[indx:]},{rev2prob[indx:]}]{char1}{char2}:{dotpar[:indx+1]}+{dotpar[-(indx+1):]} ")
                step_string = ''.join(stepbystep).strip()
                system_message = {"role": "system", "content": "You are a DNA analyzer. Please take the base pairing binaires and convert them to parens-dot-plus notation."}
                user_message = {"role": "user", "content": f"{prob_string[:len(seq1)]} {prob_string[-len(seq1):][::-1]}"}
                assistant_message = {"role": "assistant", "content": f"{step_string} ans:{dotpar}"}                
            message = {"messages": [system_message,user_message,assistant_message]}
            f.write(json.dumps(message) + '\n')

def generate_structure_jsonl(condition, seqs, output_filename):
    with open(output_filename, 'w') as f:
        for seq1, seq2, _, prob_string, dotpar in seqs:
            if condition == "naive":
                system_message = {"role": "system", "content": "You are a DNA analyzer. Please analyze the following DNA sequence pair and produce the secondary structure in parens-dot-plus notation."}
                user_message = {"role": "user", "content": f"{seq1} {seq2}"}
                assistant_message = {"role": "assistant", "content": f"{dotpar}"}
            elif condition == "rev2CoT":
                rev2 = reverse_complement(seq2)
                base_compare_string = ''.join(['1' if rev2[i] == seq1[i] else '0' for i in range(len(rev2))]) 
                pad = '_'
                zpad = "0"
                pad_seq1 = pad+seq1+pad
                pad_rev2 = pad+rev2+pad
                pad_bc = pad+base_compare_string+pad
                pad_bp1 = 3*'x'+prob_string[:len(seq1)]
                pad_bp2 = 3*'x'+prob_string[-len(seq1):][::-1]
                stepbystep =[]
                for baseind in range(len(seq1)):
                    indx = slice(baseind,baseind+3)
                    stepbystep.append(f"[{pad_seq1[indx]},{pad_rev2[indx]}]:{dotpar[:baseind+1]} ")
                step_string = ''.join(stepbystep).strip()

                system_message = {"role": "system", "content": "You are a DNA analyzer. Please analyze the following DNA sequence pair and produce the secondary structure in parens-dot-plus notation."}
                user_message = {"role": "user", "content": f"{seq1} {seq2}"}
                assistant_message = {"role": "assistant", "content": f"{rev2} {step_string} ans:{dotpar}"}    
            elif condition == "seq2CoT":
                rev2 = reverse_complement(seq2)
                pad = '_'
                pad_seq1 = pad+seq1+pad
                pad_seq2 = pad+seq2+pad
                stepbystep =[]
                for baseind in range(len(seq1)):
                    indx = slice(baseind,baseind+3)
                    stepbystep.append(f"[{pad_seq1[indx]},{pad_seq2[::-1][indx]}]:{dotpar[:baseind+1]} ")
                step_string = ''.join(stepbystep).strip()
                system_message = {"role": "system", "content": "You are a DNA analyzer. Please analyze the following DNA sequence pair and produce the secondary structure in parens-dot-plus notation."}
                user_message = {"role": "user", "content": f"{seq1} {seq2}"}
                assistant_message = {"role": "assistant", "content": f"{step_string} ans:{dotpar}"}                   
            elif condition == "+rev_comp+base_compare":
                rev2 = reverse_complement(seq2)
                base_compare_string = ''.join(['1' if rev2[i] == seq1[i] else '0' for i in range(len(rev2))])
                system_message = {"role": "system", "content": "You are a DNA analyzer. Please analyze the following DNA sequence pair and base comparison binary to produce the secondary structure in parens-dot-plus notation."}
                user_message = {"role": "user", "content": f"{seq1} {rev2} {base_compare_string}"}
                assistant_message = {"role": "assistant", "content": f"{dotpar}"}
            # elif condition == "+rev_comp+CoT":
            #     rev2 = reverse_complement(seq2)
            #     base_compare_string = ''.join(['1' if rev2[i] == seq1[i] else '0' for i in range(len(rev2))])
            #     stepbystep=[]
            #     for indx, (char1, char2) in enumerate(zip(seq1, rev2)):
            #         stepbystep.append(f"({seq1[indx:]},{rev2[indx:]}){char1}{char2}:{dotpar[indx]} ")
            #     step_string = ''.join(stepbystep).strip()
            #     system_message = {"role": "system", "content": "You are a DNA analyzer. Please analyze the following DNA sequence pair to produce the secondary structure in parens-dot-plus notation."}
            #     user_message = {"role": "user", "content": f"{seq1} {rev2}"}
            #     assistant_message = {"role": "assistant", "content": f"{step_string} {dotpar}"}   
            elif condition == "+rev_comp+CoT":
                rev2 = reverse_complement(seq2)
                base_compare_string = ''.join(['1' if rev2[i] == seq1[i] else '0' for i in range(len(rev2))]) 
                pad = '_'
                zpad = "0"
                pad_seq1 = pad+seq1+pad
                pad_rev2 = pad+rev2+pad
                pad_bc = pad+base_compare_string+pad
                pad_bp1 = 3*'x'+prob_string[:len(seq1)]
                pad_bp2 = 3*'x'+prob_string[-len(seq1):][::-1]
                stepbystep =[]
                for baseind in range(len(seq1)):
                    indx = slice(baseind,baseind+3)
                    stepbystep.append(f"[{pad_seq1[indx]},{pad_rev2[indx]}]:{dotpar[:baseind+1]} ")
                step_string = ''.join(stepbystep).strip()
                system_message = {"role": "system", "content": "You are a DNA analyzer. Please analyze the following DNA sequence pair to produce the secondary structure in parens-dot-plus notation."}
                user_message = {"role": "user", "content": f"{seq1} {rev2}"}
                assistant_message = {"role": "assistant", "content": f"{step_string} ans:{dotpar}"}                   
            message = {"messages": [system_message,user_message,assistant_message]}
            f.write(json.dumps(message) + '\n')
            #best model: ft:gpt-3.5-turbo-1106:hedilog::8qXEc0Y5

def generate_mfe_jsonl(condition, seqs, output_filename):
    with open(output_filename, 'w') as f:
        for seq1, seq2, mfe, prob_string, dotpar in seqs:
            rev2 = reverse_complement(seq2)
            if condition == "naive":
                system_message = {"role": "system", "content": "You are a DNA analyzer. Please analyze the following DNA sequence pair and determine the corresponding minimum free energy in kcal/mol."}
                user_message = {"role": "user", "content": f"{seq1} {seq2}"}
                assistant_message = {"role": "assistant", "content": f"{mfe}"}
            elif condition == "rev2CoT":
                pad = '_'
                pad_seq1 = pad+seq1+pad
                pad_rev2 = pad+rev2+pad
                stepbystep =[]
                for baseind in range(len(seq1)):
                    indx = slice(baseind,baseind+3)
                    stepbystep.append(f"[{pad_seq1[indx]},{pad_rev2[indx]}]:{dotpar[:baseind+1]} ")
                step_string = ''.join(stepbystep).strip()             
                system_message = {"role": "system", "content": "You are a DNA analyzer. Please analyze the following DNA sequence pair and determine the corresponding minimum free energy in kcal/mol."}
                user_message = {"role": "user", "content": f"{seq1} {seq2}"}
                assistant_message = {"role": "assistant", "content": f"{rev2} {step_string} ans:{mfe}"}                
            elif condition == "+rev_comp+CoT":
                pad = '_'
                pad_seq1 = pad+seq1+pad
                pad_rev2 = pad+rev2+pad
                stepbystep =[]
                for baseind in range(len(seq1)):
                    indx = slice(baseind,baseind+3)
                    stepbystep.append(f"[{pad_seq1[indx]},{pad_rev2[indx]}]:{dotpar[:baseind+1]} ")
                step_string = ''.join(stepbystep).strip()             
                system_message = {"role": "system", "content": "You are a DNA analyzer. Please analyze the following DNA sequence pair and determine the corresponding minimum free energy in kcal/mol."}
                user_message = {"role": "user", "content": f"{seq1} {rev2}"}
                assistant_message = {"role": "assistant", "content": f"{step_string} ans:{mfe}"}
            elif condition == "+rev_comp+dotpar":
                system_message = {"role": "system", "content": "You are a DNA analyzer. Please analyze the following DNA sequence pair and secondary structure to determine the corresponding minimum free energy in kcal/mol."}
                user_message = {"role": "user", "content": f"{seq1} {rev2} {dotpar}"}
                assistant_message = {"role": "assistant", "content": f"{mfe}"}                
            message = {"messages": [system_message,user_message,assistant_message]}
            f.write(json.dumps(message) + '\n')


def generate_sequence_jsonl(condition, structures, output_filename):
        with open(output_filename, 'w') as f:
            for dotpar,seq1, seq2 in structures:
                if condition == "naive":
                    system_message = {"role": "system", "content": "You are a DNA designer. Please design a pair of DNA sequences that will form the following secondary structure."}
                    user_message = {"role": "user", "content": f"{dotpar}"}
                    assistant_message = {"role": "assistant", "content": f"{seq1} {seq2}"}                    
                elif condition == "CoTrev2+rev_comp":
                    rev2 = reverse_complement(seq2)
                    pad = '_'
                    pad_dotpar = pad+dotpar[:len(seq1)]+pad
                    stepbystep =[]
                    for baseind in range(len(seq1)):
                        indx = slice(baseind,baseind+3)
                        stepbystep.append(f"[{pad_dotpar[indx]}]:[{seq1[:baseind+1]},{rev2[:baseind+1]}] ")
                    step_string = ''.join(stepbystep).strip()
                    system_message = {"role": "system", "content": "You are a DNA designer. Please design a pair of DNA sequences that will form the following secondary structure."}
                    user_message = {"role": "user", "content": f"{dotpar}"}
                    assistant_message = {"role": "assistant", "content": f"{step_string} ans:{seq1} {rev2}"}
                elif condition == "CoTseq2":
                    rev2 = reverse_complement(seq2)
                    pad = '_'
                    pad_dotpar = pad+dotpar[:len(seq1)]+pad
                    stepbystep =[]
                    for baseind in range(len(seq1)):
                        indx = slice(baseind,baseind+3)
                        stepbystep.append(f"[{pad_dotpar[indx]}]:[{seq1[:baseind+1]},{rev2[:baseind+1]}] ")
                    step_string = ''.join(stepbystep).strip()
                    system_message = {"role": "system", "content": "You are a DNA designer. Please design a pair of DNA sequences that will form the following secondary structure."}
                    user_message = {"role": "user", "content": f"{dotpar}"}
                    assistant_message = {"role": "assistant", "content": f"{step_string} ans:{seq1} {seq2}"}                                                  
                message = {"messages": [system_message,user_message,assistant_message]}
                f.write(json.dumps(message) + '\n')                

            
def run_fine_tune_job(args):
    experiment, train_size = args
    #Load training file
    training_file = client.files.create(
    file=open(f"fine_tune_sets/{experiment}_train_size_{train_size}.jsonl", "rb"),
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
    if experiment == "sequence_design":
        with open(f"training_data/structure_train_set.json", 'r') as f: 
                train_set = json.load(f)        
    else:
        with open(f"training_data/sequence_train_set.json", 'r') as f: 
                train_set = json.load(f)
    for ts in train_sizes:
        if experiment == "reverse_complement":
            generate_reverse_complement_jsonl(condition,train_set[:ts],f"fine_tune_sets/{experiment}_{condition}_train_size_{ts}.jsonl")
        elif experiment == "base_comparison":
            generate_base_comparison_jsonl(condition,train_set[:ts],f"fine_tune_sets/{experiment}_{condition}_train_size_{ts}.jsonl")
        elif experiment == "base_pairing":
            generate_base_pairing_jsonl(condition,train_set[:ts],f"fine_tune_sets/{experiment}_{condition}_train_size_{ts}.jsonl")
        elif experiment == "convert_to_dotparens":
            generate_convert_to_dotparens_jsonl(condition,train_set[:ts],f"fine_tune_sets/{experiment}_{condition}_train_size_{ts}.jsonl")
        elif experiment == "secondary_structure":
            generate_structure_jsonl(condition,train_set[:ts],f"fine_tune_sets/{experiment}_{condition}_train_size_{ts}.jsonl")
        elif experiment == "minimum_free_energy":
            generate_mfe_jsonl(condition,train_set[:ts],f"fine_tune_sets/{experiment}_{condition}_train_size_{ts}.jsonl")
        elif experiment == "sequence_design":
            generate_sequence_jsonl(condition,train_set[:ts],f"fine_tune_sets/{experiment}_{condition}_train_size_{ts}.jsonl")


    if condition is not None:
        outname = f"{experiment}_{condition}"
    else:
        outname = f"{experiment}"

    arguments = [(outname, ts) for ts in train_sizes]
    with multiprocessing.Pool(3) as pool:
        model_ids = pool.map(run_fine_tune_job, arguments)

    model_list = list(zip(train_sizes,model_ids))
    with open("model_ids/"+outname+"_models.json", 'w') as f:
        json.dump(model_list, f)



if __name__ == '__main__':
    experiment = "minimum_free_energy"
    condition = "seq2CoT"
    # train_sizes = [200, 500, 1400, 3700, 10000]
    train_sizes = [10000]
    # fine_tune(experiment,train_sizes)
    fine_tune(experiment,train_sizes,condition=condition)
    

    

    