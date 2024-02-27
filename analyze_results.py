import json
import numpy as np


def analyze_results():
    experiments = {
        "reverse_complement":{
            "naive":"naive_max_tries_20_test_size_10000"
        },
        "secondary_structure":{
            "naive":"naive_max_tries_20_test_size_10000",
            "seq2CoT":"seq2CoT_max_tries_20_test_size_10000",
            "rev2CoT":"rev2CoT_max_tries_20_test_size_10000",
            "+rev_comp+CoT":"+rev_comp+CoT_max_tries_20_test_size_10000",
            "+rev_comp_expert+CoT":"+rev_comp_expert+CoT_expert_tries_20_max_tries_20_test_size_10000",
            "+rev_comp_expert+CoT-1_try":"+rev_comp_expert+CoT_expert_tries_1_max_tries_20_test_size_10000"
        },
        "minimum_free_energy":{
            "naive":"naive_max_tries_20_test_size_10000",
            "rev2CoT":"rev2CoT_max_tries_20_test_size_10000",
            "+rev_comp+CoT":"+rev_comp+CoT_max_tries_20_test_size_10000",
            "+rev_comp_expert+CoT":"+rev_comp_expert+CoT_expert_tries_20_max_tries_20_test_size_10000",
            "+rev_comp+dotpar": "+rev_comp+dotpar_max_tries_20_test_size_10000"
        },
        "sequence_design":{
            "naive":"naive_max_tries_20_test_size_10000",
            "CoTseq2":"CoTseq2_max_tries_20_test_size_10000",
            "CoT+rev_comp":"CoTrev2+rev_comp_max_tries_20_test_size_10000",
            "CoTrev2+rev_comp_expert":"CoTrev2+rev_comp_expert_expert_tries_20_max_tries_20_test_size_10000",
            "CoTrev2+rev_comp_expert+error_checking":"CoTrev2+rev_comp_expert+error_checking+_expert_tries_20_max_tries_20_test_size_10000",
            "CoTrev2+rev_comp_expert+error_checking_expert":"CoTrev2+rev_comp_expert+error_checking_expert+_expert_tries_20_max_tries_20_test_size_10000_new"
        }
    }

    for exp, conditions in experiments.items():
        print("----")
        print(f"{exp}")
        for cond, fn in conditions.items():
            file_name = f"test_results/{exp}_{fn}.json"

            responses = []
            with open(file_name,'r') as f:
                for line in f:
                    responses.append(json.loads(line))

            if exp == "reverse_complement":
                match_count = sum(1 for entry in responses if entry["model"]== entry["rev2"])
                accuracy = match_count/len(responses)*100
                out_val = f"{accuracy=:.3g}%"
            elif exp == "secondary_structure":
                match_count = sum(1 for entry in responses if entry["model_structure"] == entry["structure"])
                # err_count = sum(1 for entry in responses if entry["model_structure"] != entry["structure"])
                # bad_form = sum(1 for entry in responses if entry["model_structure"] == "2")
          
                accuracy = match_count/len(responses)*100
                # print(bad_form/err_count*100)
                out_val = f"{accuracy=:.3g}%"                                
            elif exp == "minimum_free_energy":
                distances = [np.abs(float(entry["model_MFE"]) - float(entry["MFE"])) for entry in responses if entry["model_MFE"] != 2]
                error = np.mean(distances)
                std_dist = np.std(distances)
                out_val = f"{error=:.3g} +/- {std_dist:.3g} kcal/mol"
            elif exp == "sequence_design":
                match_count = sum(1 for entry in responses if entry["structure"] == entry["model_structure"])
                accuracy = match_count/len(responses)*100
                out_val = f"{accuracy=:.3g}%"
            print(f"{cond}:"+out_val)


if __name__ == '__main__':
    analyze_results()