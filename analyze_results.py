import json

def analyze_results(experiment,training_sizes,condition=None):
    if condition is not None:
        file_name = f'test_results/{experiment}_{condition}_test_size_{training_sizes[0]}.json'
    else:
        file_name = f'test_results/{experiment}_test_size_{training_sizes[0]}.json'

    data = []
    with open(file_name, 'r') as file:
        for line in file:
            json_obj = json.loads(line)
            data.append(json_obj)
    wrong_answers = []
    error_step = []
    for entry in data:
        if experiment == "base_comparison":
            if entry["model"] != entry["base_comparison"] and entry["model"] != "2":
                # wrong_answers.append({'ans':entry["base_comparison"],'model':entry["model"]})
                cat_seq = entry['seq1'] + entry['seq2']
                g_count = cat_seq.count('G')
                c_count = cat_seq.count('C')
                t_count = cat_seq.count('T')
                a_count = cat_seq.count('A')
                max_count = max(g_count,c_count,t_count,a_count)
                wrong_answers.append(max_count/len(cat_seq))

        elif experiment == "chain_of_experts":
            if entry["model_base_pairing"] != entry["base_pairing"]:
                wrong_answers.append((entry["model_base_pairing"],entry["base_pairing"]))
            #Identify which step is responsible for the most failures
            error_list = [entry["rev2"] != entry["model_rev2"],
            entry["model_base_comparison"] != entry["base_comparison"],
            entry["model_base_pairing"] != entry["base_pairing"],
            entry["model_structure"] != entry["structure"]] 
            if sum(error_list) != 0:
                #Find where first error occurs
                error_step.append(next(indx for indx,value in enumerate(error_list) if value == True))

    # print(f"reverse complement:{sum(eind == 0 for eind in error_step)/len(error_step)}")
    # print(f"base compare:{sum(eind == 1 for eind in error_step)/len(error_step)}")
    # print(f"base pair:{sum(eind == 2 for eind in error_step)/len(error_step)}")
    # print(f"conversion:{sum(eind == 3 for eind in error_step)/len(error_step)}")
    thing = [(mbp,ans) for mbp, ans in wrong_answers if mbp != "2"]
    print(f"{thing}")

if __name__ == '__main__':
    training_sizes = [10000]
    experiment = "chain_of_experts"
    # condition = "CoT+rev_comp"
    analyze_results(experiment,training_sizes)