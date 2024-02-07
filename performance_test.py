import json
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


def test_reverse_complement_model(sampled_sequences,retry_delay,timeout_duration, modelid):
    results = []
    with tqdm(total=len(sampled_sequences)) as pbar: 
        for seq1, seq2, mfe, prob_string, dotpar in sampled_sequences:
            bad_outputs = 0
            rev2 = reverse_complement(seq2)
            message = [
                {"role": "system", "content": "You are a DNA analyzer. Please return the reverse complement of the following sequence."},
                {"role": "user", "content": f"{seq2}"}
            ]
            while True:
                response = call_openai_api(message,timeout_duration,modelid)
                if response is not None: #in case of API timeout
                    out_string = str(response.choices[0].message.content)
                    if len(out_string) == len(seq2) and all(char in 'GCTA' for char in out_string): #verify correct output form
                        results.append({
                            "seq1": seq1,
                            "seq2": seq2,
                            "MFE": mfe,
                            "structure": dotpar,
                            "prob_string": prob_string,
                            "rev2": rev2,
                            "model": out_string   
                        })
                        break
                    else:
                        bad_outputs+=1
                        if bad_outputs == 21:
                            results.append({
                                "seq1": seq1,
                                "seq2": seq2,
                                "MFE": mfe,
                                "structure": dotpar,
                                "prob_string": prob_string,
                                "rev2": rev2,
                                "model": "2"   
                            })
                            break
                else:
                    print("timeout, retrying")
                    time.sleep(retry_delay)
            pbar.update(1)                    
    return results


def test_base_comparison_model(sampled_sequences,condition,retry_delay,timeout_duration,modelid):
    results = []
    with tqdm(total=len(sampled_sequences)) as pbar: 
        for seq1, seq2, mfe, prob_string, dotpar in sampled_sequences:
            bad_outputs = 0
            rev2 = reverse_complement(seq2)
            base_compare_string = ''.join(['1' if rev2[i] == seq1[i] else '0' for i in range(len(rev2))])
            if condition == "-rev_comp" or condition == "CoT":
                message =[{"role": "system", "content": "You are a DNA analyzer. Please compare the two partially complementary sequences and return a binary string corresponding to a valid or invalid base pairing."},
                {"role": "user", "content": f"{seq1} {seq2}"}]
            elif condition == "+rev_comp" or condition == "CoT+rev_comp":
                message = [{"role": "system", "content": "You are a DNA analyzer. Please compare the two sequences and return a binary string corresponding to characters being identical or not."},
                {"role": "user", "content": f"{seq1} {rev2}"}]

            while True:
                response = call_openai_api(message,timeout_duration,modelid)
                valid_out = False
                if response is not None: #in case of API timeout
                    out_string = str(response.choices[0].message.content)
                    if condition == "-rev_comp" or condition == "+rev_comp":
                        valid_out = len(out_string) == len(seq1) and all(char in '10' for char in out_string)
                    elif condition == "CoT" or condition == "CoT+rev_comp":
                        out_string = out_string.split("ans:")
                        if len(out_string) == 2:
                            out_string = out_string[1]
                            valid_out = len(out_string) == len(seq1) and all(char in '10' for char in out_string)

                    if valid_out:
                        results.append({
                            "seq1": seq1,
                            "seq2": seq2,
                            "MFE": mfe,
                            "structure": dotpar,
                            "prob_string": prob_string,
                            "base_comparison": base_compare_string,
                            "model": out_string   
                        })
                        break
                    else:
                        bad_outputs+=1
                        if bad_outputs == 21:
                            results.append({
                                "seq1": seq1,
                                "seq2": seq2,
                                "MFE": mfe,
                                "structure": dotpar,
                                "prob_string": prob_string,
                                "base_comparison": base_compare_string,
                                "model": "2"   
                            })
                            break
                else:
                    print("timeout, retrying")
                    time.sleep(retry_delay)
            pbar.update(1)                    
    return results                


def test_base_pairing_model(sampled_sequences,condition,retry_delay,timeout_duration,modelid):
    results = []
    with tqdm(total=len(sampled_sequences)) as pbar: 
        for seq1, seq2, mfe, prob_string, dotpar in sampled_sequences:
            bad_outputs = 0
            rev2 = reverse_complement(seq2)
            base_compare_string = ''.join(['1' if rev2[i] == seq1[i] else '0' for i in range(len(rev2))])
            if condition == "+base_comparison" or condition == "alt+base_comparison":
                message = [{"role": "system", "content": "You are a DNA analyzer. Please compare the two sequences and the corresponding base comparison binary to generate two binaries representing where base pairing occurs."},
                {"role": "user", "content": f"{seq1} {rev2} {base_compare_string}"}]
            elif condition == "-base_comparison":
                message = [{"role": "system", "content": "You are a DNA analyzer. Please compare the two sequences to generate two binaries representing where base pairing occurs."},
                {"role": "user", "content": f"{seq1} {rev2}"}]
            elif condition == "dotparens+base_comparison" or condition == "altdotparens+base_comparison" or condition[1:] == "altdotparens+base_comparison":
                message = [{"role": "system", "content": "You are a DNA analyzer. Please compare the two sequences and the corresponding base comparison binary to determine the secondary structure in dot-parens-plus notation."},
                {"role": "user", "content": f"{seq1} {rev2} {base_compare_string}"}]

            while True:
                response = call_openai_api(message,timeout_duration,modelid)
                valid_out = False
                if response is not None: #in case of API timeout
                    out_string = str(response.choices[0].message.content).split("ans:")
                    if condition == "altdotparens+base_comparison":
                        ans_string = out_string[0]
                        valid_out =  len(ans_string) == len(dotpar) and all(char in '()+.' for char in ans_string)
                        ans_string = [ans_string,'']
                    elif len(out_string) == 2:
                        if condition[1:] == "altdotparens+base_comparison":
                            ans_string = out_string[1]
                            valid_out =  len(ans_string) == len(dotpar) and all(char in '()+.' for char in ans_string)
                            ans_string = [ans_string,'']
                        else:    
                            ans_string = out_string[1].split(" ")
                            if len(ans_string) == 2 and len(ans_string[0]) == len(seq1) and len(ans_string[1]) == len(seq2):
                                if condition == "-base_comparison" or condition == "+base_comparison" or condition == "alt+base_comparison":
                                    valid_out = all(all(char in '10' for char in string) for string in ans_string)
                                elif condition == "dotparens+base_comparison":
                                    valid_out =  all(char in '(.' for char in ans_string[0]) and all(char in ').' for char in ans_string[1])

                                
                    if valid_out:
                        results.append({
                            "seq1": seq1,
                            "seq2": seq2,
                            "MFE": mfe,
                            "structure": dotpar,
                            "prob_string": prob_string,
                            "base_comparison": base_compare_string,
                            "dotpar1": dotpar[:len(seq1)],
                            "dotpar2": dotpar[-len(seq1):][::-1],
                            "bpbin1": prob_string[:len(seq1)],
                            "bpbin2": prob_string[-len(seq1):][::-1],
                            "model1": ans_string[0],
                            "model2": ans_string[1]  
                        })
                        break
                    else:
                        bad_outputs+=1
                        if bad_outputs == 21:
                            results.append({
                                "seq1": seq1,
                                "seq2": seq2,
                                "MFE": mfe,
                                "structure": dotpar,
                                "prob_string": prob_string,
                                "base_comparison": base_compare_string,
                                "dotpar1": dotpar[:len(seq1)],
                                "dotpar2": dotpar[-len(seq1):][::-1],
                                "bpbin1": prob_string[:len(seq1)],
                                "bpbin2": prob_string[-len(seq1):][::-1],
                                "model1": "2",
                                "model2": "2"  
                            })
                            break
                else:
                    print("timeout, retrying")
                    time.sleep(retry_delay)
            pbar.update(1)                    
    return results 


def test_convert_to_dotparens_model(sampled_sequences,condition,retry_delay,timeout_duration,modelid):
    results = []
    with tqdm(total=len(sampled_sequences)) as pbar: 
        for seq1, seq2, mfe, prob_string, dotpar in sampled_sequences:
            bad_outputs = 0
            rev2 = reverse_complement(seq2)
            base_compare_string = ''.join(['1' if rev2[i] == seq1[i] else '0' for i in range(len(rev2))])
            if condition == "complete" or condition == "flip+complete" or condition == "CoT":
                message = [{"role": "system", "content": "You are a DNA analyzer. Please take the base pairing binaires and convert them to parens-dot-plus notation."},
                {"role": "user", "content": f"{prob_string[:len(seq1)]} {prob_string[-len(seq1):][::-1]}"}]
            elif condition == "char_convert":
                message = [{"role": "system", "content": "You are a DNA analyzer. Please take the base pairing binaires and convert the characters to parens-dot notation."},
                {"role": "user", "content": f"{prob_string[:len(seq1)]} {prob_string[-len(seq1):][::-1]}"}]
            while True:
                response = call_openai_api(message,timeout_duration,modelid)
                valid_out = False
                if response is not None: #in case of API timeout
                    out_string = str(response.choices[0].message.content)
                    if condition == "complete":
                        ans_string = out_string
                        valid_out = len(ans_string) == len(dotpar) and all(char in '().+' for char in ans_string)
                    elif condition == "flip+complete" or "CoT":
                        out_string = out_string.split("ans:")
                        if len(out_string) == 2:
                            ans_string = out_string[1]
                            valid_out = len(ans_string) == len(dotpar) and all(char in '().+' for char in ans_string)
                    elif condition == "char_convert":
                        ans_string = out_string
                        check_string = ans_string.split(" ")
                        valid_out = (len(check_string) == 2 and len(check_string[0]) == len(check_string[1]) == len(seq1) and 
                        all(char in '(.' for char in check_string[0]) and all(char in ').' for char in check_string[1]))

                    if valid_out:
                        results.append({
                            "seq1": seq1,
                            "seq2": seq2,
                            "MFE": mfe,
                            "structure": dotpar,
                            "prob_string": prob_string,
                            "base_comparison": base_compare_string,
                            "split_parensdot": dotpar[:len(seq1)]+" "+dotpar[-len(seq1):][::-1],
                            "model": ans_string   
                        })
                        break
                    else:
                        bad_outputs+=1
                        if bad_outputs == 21:
                            results.append({
                                "seq1": seq1,
                                "seq2": seq2,
                                "MFE": mfe,
                                "structure": dotpar,
                                "prob_string": prob_string,
                                "base_comparison": base_compare_string,
                                "split_parensdot": dotpar[:len(seq1)]+" "+dotpar[-len(seq1):][::-1],
                                "model": "2"   
                            })
                            break
                else:
                    print("timeout, retrying")
                    time.sleep(retry_delay)
            pbar.update(1)                    
    return results    

def test_chain_of_experts(sampled_sequences,condition,coe_args,retry_delay,timeout_duration):    
    results = []
    if condition == "no_error_check":
        max_tries = 1
    else:
        max_tries = 21
    with tqdm(total=len(sampled_sequences)) as pbar: 
        for seq1, seq2, mfe, prob_string, dotpar in sampled_sequences:
            #reverse complement
            bad_outputs = 0
            message = [
                {"role": "system", "content": "You are a DNA analyzer. Please return the reverse complement of the following sequence."},
                {"role": "user", "content": f"{seq2}"}
            ]
            while True:
                response = call_openai_api(message,timeout_duration,coe_args["model_list"][0])
                if response is not None: #in case of API timeout
                    out_string = str(response.choices[0].message.content)
                    if len(out_string) == len(seq2) and all(char in 'GCTA' for char in out_string): #verify correct output form
                        model_rev2 = out_string
                        break
                    else:
                        bad_outputs += 1
                        if bad_outputs == max_tries:
                            model_rev2 = "2"
                            break
                else:
                    print("timeout on reverse complement, retrying")
                    time.sleep(retry_delay)

            if model_rev2 != "2":
                #base comparison
                bad_outputs = 0
                valid_out = False
                message = [{"role": "system", "content": "You are a DNA analyzer. Please compare the two sequences and return a binary string corresponding to characters being identical or not."},
                        {"role": "user", "content": f"{seq1} {model_rev2}"}]
                while True:
                    response = call_openai_api(message,timeout_duration,coe_args["model_list"][1])
                    if response is not None: #in case of API timeout
                        out_string = str(response.choices[0].message.content).split("ans:")
                        if len(out_string) == 2:
                            ans_string = out_string[1]
                            valid_out = len(ans_string) == len(seq1) and all(char in '10' for char in ans_string)
                        if valid_out:
                            model_base_comparison = ans_string
                            break
                        else:
                            bad_outputs += 1
                            if bad_outputs == max_tries:
                                model_base_comparison = "2"
                                break
                    else:
                        print("timeout on base comparison, retrying")
                        time.sleep(retry_delay)
            else:
                model_base_comparison = "2"
            
            if model_base_comparison != "2":
                #base pairing
                bad_outputs = 0
                valid_out = False
                message = [{"role": "system", "content": "You are a DNA analyzer. Please compare the two sequences and the corresponding base comparison binary to generate two binaries representing where base pairing occurs."},
                        {"role": "user", "content": f"{seq1} {model_rev2} {model_base_comparison}"}]
                while True:
                    response = call_openai_api(message,timeout_duration,coe_args["model_list"][2])
                    if response is not None: #in case of API timeout
                        out_string = str(response.choices[0].message.content).split("ans:")
                        if len(out_string) == 2:
                            ans_string = out_string[1].split(" ")
                            valid_out = len(ans_string) == 2 and len(ans_string[0]) == len(seq1) and len(ans_string[1]) == len(seq2) and all(all(char in '10' for char in string) for string in ans_string)
                        if valid_out:
                            model_base_pairing = out_string[1]
                            break
                        else:
                            bad_outputs += 1
                            if bad_outputs == max_tries:
                                model_base_pairing = "2"
                                break
                    else:
                        print("timeout on base pairing, retrying")
                        time.sleep(retry_delay)
            else: 
                model_base_pairing = "2"

            if model_base_pairing != "2":
                #convert to parens-dot-plus
                bad_outputs = 0
                message = [{"role": "system", "content": "You are a DNA analyzer. Please take the base pairing binaires and convert them to parens-dot-plus notation."},
                {"role": "user", "content": f"{model_base_pairing}"}]
                while True:
                    response = call_openai_api(message,timeout_duration,coe_args["model_list"][3])                
                    if response is not None: #in case of API timeout
                        out_string = str(response.choices[0].message.content)
                        if len(out_string) == len(seq1)*2+1 and all(char in '().+' for char in out_string):
                            model_structure = out_string
                            break
                        else:
                            bad_outputs += 1
                            if bad_outputs == max_tries:
                                model_structure = "2"
                                break
                    else:
                        print("timeout on parens-dot-plus conversion, retrying")
                        time.sleep(retry_delay)
            else:
                model_structure = "2"

            rev2 = reverse_complement(seq2)
            base_compare_string = ''.join(['1' if rev2[i] == seq1[i] else '0' for i in range(len(rev2))])            
            results.append({
                "seq1": seq1,
                "seq2": seq2,
                "MFE": mfe,
                "structure": dotpar,
                "prob_string": prob_string,
                "rev2": rev2,
                "base_comparison": base_compare_string,
                "base_pairing": prob_string[:len(seq1)]+" "+prob_string[-len(seq2):][::-1],
                "model_rev2": model_rev2,
                "model_base_comparison": model_base_comparison,
                "model_base_pairing": model_base_pairing,
                "model_structure": model_structure            
                })
            pbar.update(1)
    return results                    


def test_secondary_structure_model(sampled_sequences,condition,retry_delay,timeout_duration,modelid):
    results = []
    if condition == "chain_of_thought_no_error_check":
        max_tries = 1
    else:
        max_tries = 21

    with tqdm(total=len(sampled_sequences)) as pbar: 
        for seq1, seq2, mfe, prob_string, dotpar in sampled_sequences:
            bad_outputs = 0
            rev2 = reverse_complement(seq2)
            base_compare_string = ''.join(['1' if rev2[i] == seq1[i] else '0' for i in range(len(rev2))])
            message = [{"role": "system", "content": "You are a DNA analyzer. Please take the following DNA sequence pair and produce the secondary structure in parens-dot-plus notation."},
            {"role": "user", "content": f"{seq1} {seq2}"}]
            while True:
                response = call_openai_api(message,timeout_duration,modelid)
                valid_out = False
                if response is not None: #in case of API timeout
                    out_string = str(response.choices[0].message.content)  
                    if condition == "naive":
                        ans_string = out_string
                        valid_out = len(ans_string) == len(dotpar) and all(char in '().+' for char in ans_string) 
                    elif condition == "chain_of_thought" or condition == "chain_of_thought_no_error_check":
                        out_string = out_string.split("ans:")
                        if len(out_string) == 2:
                            ans_string = out_string[1]
                            valid_out = len(ans_string) == len(dotpar) and all(char in '().+' for char in ans_string)

                    if valid_out:
                        results.append({
                            "seq1": seq1,
                            "seq2": seq2,
                            "MFE": mfe,
                            "structure": dotpar,
                            "prob_string": prob_string,
                            "base_comparison": base_compare_string,
                            "model_structure": ans_string   
                        })
                        break
                    else:
                        bad_outputs+=1
                        if bad_outputs == max_tries:
                            results.append({
                                "seq1": seq1,
                                "seq2": seq2,
                                "MFE": mfe,
                                "structure": dotpar,
                                "prob_string": prob_string,
                                "base_comparison": base_compare_string,
                                "model_structure": "2"   
                            })
                            break
                else:
                    print("timeout, retrying")
                    time.sleep(retry_delay)
            pbar.update(1)                    
    return results    
                    
                        

def analyze_model(experiment,condition, train_size, modelid=None, coe_args=None):
    retry_delay = 5  # Delay in seconds between retries
    timeout_duration = 180  # Timeout in seconds for each API call

    with open(f"training_data/DNA_sequence_validation_set.json", 'r') as f:
        val_set = json.load(f)

    print("starting analysis")
    if experiment == "reverse_complement":
        responses = test_reverse_complement_model(val_set,retry_delay,timeout_duration,modelid)
    elif experiment == "base_comparison":
        responses = test_base_comparison_model(val_set,condition,retry_delay,timeout_duration,modelid)
    elif experiment == "base_pairing":
        responses = test_base_pairing_model(val_set,condition,retry_delay,timeout_duration,modelid)
    elif experiment == "convert_to_dotparens":
        responses = test_convert_to_dotparens_model(val_set,condition,retry_delay,timeout_duration,modelid)
    elif experiment == "chain_of_experts":
        responses = test_chain_of_experts(val_set,condition,coe_args,retry_delay,timeout_duration)
    elif experiment == "secondary_structure":
        responses = test_secondary_structure_model(val_set,condition,retry_delay,timeout_duration,modelid)

    if condition is not None:
        val_model_out_filename = f"test_results/{experiment}_{condition}_test_size_{train_size}.json"
    else:
        val_model_out_filename = f"test_results/{experiment}_test_size_{train_size}.json"

    # Write the results to a JSON file
    with open(val_model_out_filename, 'w') as f:
        for item in responses:
            f.write(json.dumps(item) + '\n')

    # responses = []
    # with open(val_model_out_filename,'r') as f:
    #     for line in f:
    #         responses.append(json.loads(line))


    # Calculate the number of matches and the total count
    if experiment == "reverse_complement":
        match_count = sum(1 for entry in responses if entry["model"]== entry["rev2"])
    elif experiment == "base_comparison":
        match_count = sum(1 for entry in responses if entry["model"]== entry["base_comparison"])
    elif experiment == "base_pairing":
        if condition == "-base_comparison" or condition == "+base_comparison" or condition == "alt+base_comparison":
            match_count = sum(1 for entry in responses if entry["model1"]== entry["bpbin1"] and entry["model2"]== entry["bpbin2"])
            bad_out = sum(1 for entry in responses if entry["model1"] == "2")
        elif condition == "altdotparens+base_comparison" or condition[1:] == "altdotparens+base_comparison":
             match_count = sum(1 for entry in responses if entry["model1"]== entry["structure"])  
    elif experiment == "convert_to_dotparens":
        if condition == "complete" or condition == "flip+complete" or condition == "CoT":
            match_count = sum(1 for entry in responses if entry["model"]== entry["structure"])
            bad_out = sum(1 for entry in responses if entry["model"]== "2")
        elif condition == "char_convert":
            match_count = sum(1 for entry in responses if entry["model"]== entry["split_parensdot"])
    elif experiment == "chain_of_experts" or "secondary_structure":
        match_count = sum(1 for entry in responses if entry["model_structure"] == entry["structure"])
    total_count = len(val_set)

    # # Calculate the accuracy
    match_percentage = (match_count / total_count) * 100
    print(f"Model accuracy: {match_percentage}%")
    # print(bad_out/total_count*100)



def performance_test(experiment,condition=None):
    if experiment == "chain_of_experts":
        training_size = 10000
        experiment_list = ["reverse_complement","base_comparison","base_pairing","convert_to_dotparens"]
        condition_list = [None,"CoT+rev_comp","+base_comparison","complete"]
        name_list = []
        for exp, cond in zip(experiment_list, condition_list):
            if cond is not None:
                name_list.append(f"model_ids/{exp}_{cond}_models.json")
            else:
                name_list.append(f"model_ids/{exp}_models.json")
        model_list = []
        for fn in name_list:
            with open(fn,'r') as f:
                model_info = json.load(f)[0]
                if model_info[0] != training_size:
                    raise ValueError(f"Expected model training size {training_size}, but got {model_info[0]}")
                model_list.append(model_info[1])
        coe_args = {"experiment_list": experiment_list,"condition_list": condition_list,"model_list":model_list}        
        analyze_model(experiment,condition,training_size, coe_args=coe_args)

    else:        
        if condition is not None:
            file_name =  f"model_ids/{experiment}_{condition}_models.json"
        else:
            file_name =  f"model_ids/{experiment}_models.json"

        with open(file_name,'r') as f:
            model_list = json.load(f) 
        for ts, model_id in model_list:
            analyze_model(experiment,condition,ts, modelid=model_id)


if __name__ == '__main__':
    experiment = "secondary_structure"
    condition = "chain_of_thought_no_error_check"
    performance_test(experiment,condition=condition)
    # performance_test(experiment)