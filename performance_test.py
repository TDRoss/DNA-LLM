import json
import re
import numpy as np
from tqdm import tqdm
import nupack as nup
import time
import concurrent
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
client = OpenAI()

def reverse_complement(dna):
    """Return the reverse complement of a DNA sequence."""
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    return ''.join(complement[base] for base in reversed(dna))

def is_float(s):
    try:
        float(s) 
        return True
    except ValueError:
        return False 


def structure_from_strands(strand1, strand2, nupackmodel):
    A = nup.Strand(strand1, name='A')
    B = nup.Strand(strand2, name='B')
    c1 = nup.Complex([A,B]) 
    complex_set = nup.ComplexSet(strands={A: 1e-8, B: 1e-8}, complexes=nup.SetSpec(max_size=0, include=[c1]))
    complex_analysis = nup.complex_analysis(complex_set, compute=['mfe','pfunc', 'pairs'], model=nupackmodel)
    complex_vals = complex_analysis[c1]
    structure = str(complex_vals.mfe[0].structure)
    return structure    


def call_openai_api(message, timeout_duration, modelid):
    with ThreadPoolExecutor() as executor:
        future = executor.submit(client.chat.completions.create, 
            model = modelid,
            messages = message,
        )
        try:
            return future.result(timeout=timeout_duration)
        except concurrent.futures.TimeoutError:
            print(f"API call timed out for message: {message}")
            return None


def test_reverse_complement_model(sampled_sequences,condition,max_tries,retry_delay,timeout_duration, modelid):
    results = []
    with tqdm(total=len(sampled_sequences),leave=False) as pbar: 
        for seq1, seq2, mfe, prob_string, dotpar in sampled_sequences:
            bad_outputs = 0
            rev2 = reverse_complement(seq2)
            res_dic = {     "seq1": seq1,
                            "seq2": seq2,
                            "MFE": mfe,
                            "structure": dotpar,
                            "prob_string": prob_string,
                            "rev2": rev2,
                        }            
            message = [
                {"role": "system", "content": "You are a DNA analyzer. Please return the reverse complement of the following sequence."},
                {"role": "user", "content": f"{seq2}"}
            ]
            while True:
                response = call_openai_api(message,timeout_duration,modelid)
                if response is not None: #in case of API timeout
                    out_string = str(response.choices[0].message.content)
                    if condition == "naive":
                        ans_string = out_string
                        valid_out = len(ans_string) == len(seq2) and all(char in 'GCTA' for char in ans_string)
                    elif condition == "CoT" and "ans:" in out_string:
                        ans_string = out_string.split("ans:")[1]
                        valid_out = len(ans_string) == len(seq2) and all(char in 'GCTA' for char in ans_string)

                    if valid_out: #verify correct output form
                        res_dic["model"] = ans_string
                        results.append(res_dic)
                        break
                    else:
                        bad_outputs+=1
                        if bad_outputs == max_tries:
                            res_dic["model"] = "2"
                            results.append(res_dic)
                            break
                else:
                    print("timeout, retrying")
                    time.sleep(retry_delay)
            pbar.update(1)                    
    return results



def test_secondary_structure_model(sampled_sequences,condition,max_tries,retry_delay,timeout_duration,modelid,coe_args=None):
    results = []
    if coe_args:
        max_tries_rev_comp = coe_args["max_tries"]
        modelid_rev_comp = coe_args["modelid_rev_comp"]
    with tqdm(total=len(sampled_sequences),leave=False) as pbar: 
        for seq1, seq2, mfe, prob_string, dotpar in sampled_sequences:
            bad_outputs = 0
            rev2 = reverse_complement(seq2)
            base_compare_string = ''.join(['1' if rev2[i] == seq1[i] else '0' for i in range(len(rev2))])
            res_dic = {     "seq1": seq1,
                            "seq2": seq2,
                            "MFE": mfe,
                            "structure": dotpar,
                            "prob_string": prob_string,
                            "base_comparison": base_compare_string,
                        }
            if condition == "naive" or condition == "rev2CoT" or condition == "seq2CoT":
                message = [{"role": "system", "content": "You are a DNA analyzer. Please analyze the following DNA sequence pair and produce the secondary structure in parens-dot-plus notation."},
                {"role": "user", "content": f"{seq1} {seq2}"}]
            elif condition == "+rev_comp+base_compare":
                message = [{"role": "system", "content": "You are a DNA analyzer. Please analyze the following DNA sequence pair and base comparison binary to produce the secondary structure in parens-dot-plus notation."},
                {"role": "user", "content": f"{seq1} {rev2} {base_compare_string}"}]
            elif condition == "+rev_comp+CoT":
                message = [{"role": "system", "content": "You are a DNA analyzer. Please analyze the following DNA sequence pair to produce the secondary structure in parens-dot-plus notation."},
                {"role": "user", "content": f"{seq1} {rev2}"}]
            elif condition == "+rev_comp+base_compare+CoT":
                message = [{"role": "system", "content": "You are a DNA analyzer. Please analyze the following DNA sequence pair and base comparison binary to produce the secondary structure in parens-dot-plus notation."},
                {"role": "user", "content": f"{seq1} {rev2} {base_compare_string}"}]
            elif condition == "CoT_error_check":
                message = [{"role": "system", "content": "You are a DNA analyzer. Please analyze the following DNA sequence pair to produce the secondary structure in parens-dot-plus notation."},
                {"role": "user", "content": f"{seq1} {seq2}"}]                
            elif "+rev_comp_expert+CoT" in condition:
                rev_comp_res = test_reverse_complement_model([(seq1, seq2, mfe, prob_string, dotpar)],"naive",max_tries_rev_comp,retry_delay,timeout_duration,modelid_rev_comp)
                model_rev2 = rev_comp_res[0]["model"]
                message = [{"role": "system", "content": "You are a DNA analyzer. Please analyze the following DNA sequence pair to produce the secondary structure in parens-dot-plus notation."},
                {"role": "user", "content": f"{seq1} {model_rev2}"}]
            while True:
                response = call_openai_api(message,timeout_duration,modelid)
                valid_out = False
                if response is not None: #in case of API timeout
                    out_string = str(response.choices[0].message.content)  
                    if condition == "naive" or condition == "+rev_comp+base_compare":
                        ans_string = out_string
                        valid_out = len(ans_string) == len(seq1)+len(seq2)+1 and all(char in '().+' for char in ans_string) 
                    elif "CoT" in condition: 
                        out_string = out_string.split("ans:")
                        if len(out_string) == 2:
                            ans_string = out_string[1]
                            valid_out = len(ans_string) == len(seq1)+len(seq2)+1 and all(char in '().+' for char in ans_string)
                    if valid_out:
                        res_dic.update({"model_structure": ans_string})
                        if condition == "+rev_comp_expert+CoT":
                            res_dic.update({"model_rev2": model_rev2})
                        results.append(res_dic)
                        break
                    else:
                        bad_outputs+=1
                        if bad_outputs == max_tries:
                            res_dic.update({"model_structure": "2"})
                            if condition == "+rev_comp_expert+CoT":
                                res_dic.update({"model_rev2": model_rev2})
                            results.append(res_dic)
                            break                            
                else:
                    print("timeout, retrying")
                    time.sleep(retry_delay)
            pbar.update(1)                    
    return results    

def test_mfe_model(sampled_sequences,condition,max_tries,retry_delay,timeout_duration,modelid,coe_args=None):
    results = []
    if coe_args:
        max_tries_rev_comp = coe_args["max_tries"]
        modelid_rev_comp = coe_args["modelid_rev_comp"]    
    with tqdm(total=len(sampled_sequences)) as pbar: 
        for seq1, seq2, mfe, prob_string, dotpar in sampled_sequences:
            bad_outputs = 0
            rev2 = reverse_complement(seq2)
            res_dic = {
                "seq1": seq1,
                "seq2": seq2,
                "MFE": mfe,
                "structure": dotpar,
                "prob_string": prob_string,                
            }
            if condition == "naive" or condition == "rev2CoT":  
                message = [{"role": "system", "content": "You are a DNA analyzer. Please analyze the following DNA sequence pair and determine the corresponding minimum free energy in kcal/mol."},
                {"role": "user", "content": f"{seq1} {seq2}"}]
            elif condition == "+rev_comp+CoT":
                message = [{"role": "system", "content": "You are a DNA analyzer. Please analyze the following DNA sequence pair and determine the corresponding minimum free energy in kcal/mol."},
                {"role": "user", "content": f"{seq1} {rev2}"}]
            elif "+rev_comp_expert+CoT" in condition:
                rev_comp_res = test_reverse_complement_model([(seq1, seq2, mfe, prob_string, dotpar)],"naive",max_tries_rev_comp,retry_delay,timeout_duration,modelid_rev_comp)
                model_rev2 = rev_comp_res[0]["model"]                
                message = [{"role": "system", "content": "You are a DNA analyzer. Please analyze the following DNA sequence pair and determine the corresponding minimum free energy in kcal/mol."},
                {"role": "user", "content": f"{seq1} {model_rev2}"}]                                
            elif condition == "+rev_comp+dotpar":
                message = [{"role": "system", "content": "You are a DNA analyzer. Please analyze the following DNA sequence pair and secondary structure to determine the corresponding minimum free energy in kcal/mol."},
                {"role": "user", "content": f"{seq1} {rev2} {dotpar}"}]                
            while True:
                response = call_openai_api(message,timeout_duration,modelid)
                valid_out = False
                if response is not None: #in case of API timeout
                    out_string = str(response.choices[0].message.content)
                    if condition == "naive" or condition == "+rev_comp+dotpar":
                        ans_string = out_string
                        valid_out = is_float(ans_string) and "-" in ans_string
                    elif "CoT" in condition and "ans:" in out_string:
                        ans_string = out_string.split("ans:")[1]
                        valid_out = is_float(ans_string) and "-" in ans_string

                    if valid_out:
                        res_dic.update({"model_MFE": ans_string })
                        if condition =="+rev_comp_expert+CoT":
                            res_dic.update({"model_rev2": model_rev2})
                        results.append(res_dic)
                        break
                    else:
                        bad_outputs+=1
                        if bad_outputs == max_tries:
                            res_dic.update({"model_MFE": "2" })
                            if condition =="+rev_comp_expert+CoT":
                                res_dic.update({"model_rev2": "2"})                            
                            results.append(res_dic)
                            break
                else:
                    print("timeout, retrying")
                    time.sleep(retry_delay)
            pbar.update(1)                    
    return results    


def test_sequence_model(structures,condition,max_tries,retry_delay,timeout_duration,modelid,coe_args=None):
    if coe_args:
        max_tries_rev_comp = coe_args["max_tries"]        
        if "+rev_comp_expert" in condition:
            modelid_rev_comp = coe_args["modelid_rev_comp"]
        if "+error_checking_expert+" in condition:
            modelid_dotpar = coe_args["modelid_dotpar"]

    nupackmodel = nup.Model(material='DNA',celsius=20)      
    results = []
    with tqdm(total=len(structures)) as pbar: 
        for dotpar,_,_ in structures:
            bad_outputs = 0

            message = [{"role": "system", "content": "You are a DNA designer. Please design a pair of DNA sequences that will form the following secondary structure."},
            {"role": "user", "content": f"{dotpar}"}]
            while True:
                response = call_openai_api(message,timeout_duration,modelid)
                valid_out = False
                if response is not None: #in case of API timeout
                    out_string = str(response.choices[0].message.content)
                    if condition == "naive":
                        ans_string = out_string.split(" ")
                        valid_out = len(ans_string) == 2 and len(ans_string[0]) == len(ans_string[1]) == (len(dotpar)-1)/2 and all(all(char in 'GCTA' for char in string) for string in ans_string)
                    elif "CoT" in condition and "ans:" in out_string:
                        ans_string = out_string.split("ans:")[1]
                        ans_string = ans_string.split(" ")
                        valid_out = len(ans_string) == 2 and len(ans_string[0]) == len(ans_string[1]) == (len(dotpar)-1)/2 and all(all(char in 'GCTA' for char in string) for string in ans_string)
                    if valid_out:
                        model_seq1 = ans_string[0]
                        if condition == "CoTrev2+rev_comp":
                            model_seq2 = reverse_complement(ans_string[1])
                        elif "CoTrev2+rev_comp_expert" in condition and "+error_checking+" not in condition:
                            rev_comp_res = test_reverse_complement_model([("2", ans_string[1], "2", "2", "2")],"naive",max_tries_rev_comp,retry_delay,timeout_duration,modelid_rev_comp)
                            model_seq2 = rev_comp_res[0]["model"]
                            if model_seq2 == "2":
                                valid_out = False
                        elif valid_out and (condition == "CoTseq2" or condition == "naive"):
                            model_seq2 = ans_string[1]
                    
                    if valid_out and "+error_checking+" in condition:
                        model_dotpar = structure_from_strands(model_seq1,model_seq2,nupackmodel)
                        valid_out = model_dotpar == dotpar
                    
                    elif valid_out and "+error_checking_expert+" in condition:
                            dotpar_res = test_secondary_structure_model([(model_seq1, ans_string[1], "2", "2", dotpar)],"CoT_error_check",max_tries_rev_comp,retry_delay,timeout_duration,modelid_dotpar)
                            expert_dotpar = dotpar_res[0]["model_structure"]
                            valid_out =  expert_dotpar == dotpar
                            if valid_out:
                                rev_comp_res = test_reverse_complement_model([("2", ans_string[1], "2", "2", "2")],"naive",max_tries_rev_comp,retry_delay,timeout_duration,modelid_rev_comp)
                                model_seq2 = rev_comp_res[0]["model"]
                                if valid_out and model_seq2 !="2":
                                    model_dotpar = structure_from_strands(model_seq1,model_seq2,nupackmodel)                                                       
                                else:
                                    valid_out = False
                            

                    if valid_out:
                        if "error_checking" not in condition:
                            model_dotpar = structure_from_strands(model_seq1,model_seq2,nupackmodel)
                        res_dic={
                        "structure": dotpar,
                        "model_structure": model_dotpar,
                        "model_seq1": model_seq1,
                        "model_seq2": model_seq2                                 
                        }
                        if "+error_checking_expert+" in condition:
                            res_dic["expert_dotpar"] = expert_dotpar
                        results.append(res_dic)
                        break
                    else:
                        bad_outputs+=1
                        if bad_outputs == max_tries:
                            res_dic={
                            "structure": dotpar,
                            "model_structure": "2",
                            "model_seq1": "2",
                            "model_seq2": "2"                                 
                            }
                            if "+error_checking_expert+" in condition:
                                res_dic["expert_dotpar"] = "2"
                            results.append(res_dic)
                            break
                else:
                    print("timeout, retrying")
                    time.sleep(retry_delay)
            pbar.update(1)                    
    return results    
                            

def analyze_model(experiment,condition, train_size, max_tries, modelid=None, coe_args=None):
    retry_delay = 5  # Delay in seconds between retries
    timeout_duration = 180  # Timeout in seconds for each API call

    if experiment == "sequence_design":
        with open(f"training_data/structure_validation_set.json", 'r') as f:
            val_set = json.load(f)        
    else:    
        with open(f"training_data/sequence_validation_set.json", 'r') as f:
            val_set = json.load(f)

    print("starting analysis")
    if experiment == "reverse_complement":
        responses = test_reverse_complement_model(val_set,condition,max_tries,retry_delay,timeout_duration,modelid)
    elif experiment == "secondary_structure":
        responses = test_secondary_structure_model(val_set,condition,max_tries,retry_delay,timeout_duration,modelid,coe_args=coe_args)
    elif experiment == "minimum_free_energy":
        responses = test_mfe_model(val_set,condition,max_tries,retry_delay,timeout_duration,modelid,coe_args=coe_args)
    elif experiment == "sequence_design":
        responses = test_sequence_model(val_set,condition,max_tries,retry_delay,timeout_duration,modelid,coe_args=coe_args)

    if condition is not None:
        val_model_out_filename = f"test_results/{experiment}_{condition}_max_tries_{max_tries}_test_size_{train_size}.json"
    else:
        val_model_out_filename = f"test_results/{experiment}_max_tries_{max_tries}_test_size_{train_size}.json"

    # Write the results to a JSON file
    with open(val_model_out_filename, 'w') as f:
        for item in responses:
            f.write(json.dumps(item) + '\n')

    # responses = []
    # with open(val_model_out_filename,'r') as f:
    #     for line in f:
    #         responses.append(json.loads(line))


    # Calculate the number of matches and the total count
    # if experiment == "reverse_complement":
    #     match_count = sum(1 for entry in responses if entry["model"]== entry["rev2"])
    # elif experiment == "secondary_structure":
    #     match_count = sum(1 for entry in responses if entry["model_structure"] == entry["structure"])
    # elif experiment == "minimum_free_energy":
    #     error = np.mean([np.abs(float(entry["model_MFE"]) - float(entry["MFE"])) for entry in responses if entry["model_MFE"] != 2])
    #     print(f"{error=}")
    # elif experiment == "sequence_design":
    #     match_count = sum(1 for entry in responses if entry["structure"] == entry["model_structure"])

    # if experiment != "minimum_free_energy":
    #     total_count = len(val_set)

    #     # # Calculate the accuracy
    #     match_percentage = (match_count / total_count) * 100
    #     print(f"Model accuracy: {match_percentage}%")
    #     # print(bad_out/total_count*100)


def performance_test(experiment,max_tries,condition=None):
    if (experiment == "secondary_structure" and "+rev_comp_expert+CoT" in condition) or \
        (experiment == "minimum_free_energy" and "+rev_comp_expert+CoT" in condition) or \
        (experiment == "sequence_design" and "CoTrev2+rev_comp_expert" in condition):

        substrings = ["+rev_comp_expert+CoT","+rev_comp_expert+CoT","CoTrev2+rev_comp_expert"]
        for subs in substrings:
            if subs in condition:
                subcondition = subs.replace("_expert","")
                break
        with open(f"model_ids/{experiment}_{subcondition}_models_ts.json",'r') as f:
            model_list = json.load(f)
        
        for indx, (train_size, modelid) in enumerate(model_list):
            coe_args = {}
            with open("model_ids/reverse_complement_naive_models_ts.json",'r') as f:
                ts, coe_args["modelid_rev_comp"] = json.load(f)[indx]
            if ts != train_size:
                raise ValueError("training sizes are not equal!")
            if "+error_checking_expert+" in condition:
                with open("model_ids/secondary_structure_+rev_comp+CoT_models_ts.json",'r') as f:
                    ts, coe_args["modelid_dotpar"] = json.load(f)[indx]
                if ts != train_size:
                    raise ValueError("training sizes are not equal!")                 

            match = re.search(r'\d+$', condition)
            coe_args["max_tries"] = int(match.group())
            if train_size > 1401:
                analyze_model(experiment,condition,train_size,max_tries,modelid=modelid,coe_args=coe_args)
    else:        
        if condition is not None:
            file_name =  f"model_ids/{experiment}_{condition}_models_ts.json"
        else:
            file_name =  f"model_ids/{experiment}_models.json"

        with open(file_name,'r') as f:
            model_list = json.load(f) 
        for ts, model_id in model_list:
            analyze_model(experiment,condition,ts,max_tries, modelid=model_id)


if __name__ == '__main__':
    experiment = "sequence_design"
    condition = "+CoTrev2+rev_comp_expert+error_checking_expert+_expert_tries_3"
    max_tries = 3
    performance_test(experiment,max_tries,condition=condition,)
    # performance_test(experiment,max_tries)