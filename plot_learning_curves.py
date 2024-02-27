import json
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle

import numpy as np

train_sizes = [200, 500, 1400, 3700, 10000]
color_list = [(102.0/255,194.0/255,165.0/255),(252.0/255,141.0/255,98.0/255)]
# m = MarkerStyle("D")
# m._transform.scale(1.0, 0.6)
# marker_list = ['d',m]
experiments =[
    {"names": ["reverse complement","secondary structure"], 
  "exp_base": ["reverse_complement_naive_max_tries_20","secondary_structure_+rev_comp_expert+CoT_expert_tries_20_max_tries_20"]},
  {"names": ["minimum free energy"],
  "exp_base": ["minimum_free_energy_+rev_comp_expert+CoT_expert_tries_20_max_tries_20"]},
  {"names": ["sequence design"],
  "exp_base":["sequence_design_+CoTrev2+rev_comp_expert+error_checking_expert+_expert_tries_3_max_tries_3"]}
]

for exp in experiments:
    if len(exp["names"]) > 1:
        plt_labels = exp["names"]
    else:
        plt_labels = None
    for ind, (name, base) in enumerate(zip(exp["names"],exp["exp_base"])):
        file_names = [f"test_results/{base}_test_size_{tsz}.json" for tsz in train_sizes]
        results = []
        for fn in file_names:
            data = []
            with open(fn, 'r') as file:
                for line in file:
                    data.append(json.loads(line))
            if name == "reverse complement":
                match_count = sum(1 for entry in data if entry["model"]== entry["rev2"])
                accuracy = match_count/len(data)*100
            elif name == "secondary structure":
                match_count = sum(1 for entry in data if entry["model_structure"] == entry["structure"])
                accuracy = match_count/len(data)*100
            elif name == "minimum free energy":
                accuracy = [np.abs(float(entry["model_MFE"]) - float(entry["MFE"])) for entry in data if entry["model_MFE"] != 2]
            elif name == "sequence design":
                match_count = sum(1 for entry in data if entry["structure"] == entry["model_structure"])
                accuracy = match_count/len(data)*100
            results.append(accuracy)

        if name == "minimum free energy":
            # mae, stdev = zip(*results)
            # mae, stdev = list(mae), list(stdev)
            plt.boxplot(results,showfliers=False)
            # plt.errorbar(train_sizes,mae,yerr=stdev,color=color_list[ind],label=(plt_labels[ind] if plt_labels else None), fmt='o', capsize=5, capthick=2, ecolor='black', markeredgecolor='black')
        else:
            plt.scatter(train_sizes,results,color=color_list[ind],label=(plt_labels[ind] if plt_labels else None),s=300,edgecolor='black', linewidths=1)


    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel("training size", fontsize=18)

    if exp["names"][0] == "minimum free energy":
        plt.ylabel("absolute error (kcal/mol)", fontsize=18)
        plt.xticks([1, 2, 3, 4, 5], ['200', '500','1400', '3700', '10000'])
        # plt.ylim(0,5.7)
    else:
        plt.xlim(0,10500)    
        plt.ylabel("accuracy %", fontsize=18)
        plt.ylim(0,105)
    ax.tick_params(axis='both', which='major', labelsize=18)
    if len(exp["names"]) == 2:
        handles, labels = plt.gca().get_legend_handles_labels()
        order = [1,0]  # Desired order (indices of datasets)        
        plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order],loc='upper left', bbox_to_anchor=(1, 1), fontsize=18)
    # plt.subplots_adjust(right=0.7)
    # plt.margins(x=0.1, y=0.1)
    plt.savefig(f"learning_curve_{exp['names'][0]}.pdf",format='pdf', bbox_inches='tight')
    plt.clf()

                

# plt_labels = ["reverse complement", "base comparison"]
# for ind, exps in enumerate(experiments):
#     file_names = [f"data/{exps}_{tsz}.json" for tsz in train_sizes]
#     results = []
#     for fn in file_names:
#         data = []
#         with open(fn, 'r') as file:
#             for line in file:
#                 json_obj = json.loads(line)
#                 data.append(json_obj)
#         if ind == 0: 
#             match_count = sum(1 for entry in data if entry["model"]== entry["rev-comp"])
#         else:
#             match_count = sum(1 for entry in data if entry["model"]== entry["matching"])

#         total_count = len(data)
#         accuracy = (match_count/total_count) * 100
#         results.append(accuracy)
#     plt.scatter(train_sizes,results,color=color_list[ind],label=plt_labels[ind],marker=marker_list[ind],s=300,edgecolor='black', linewidths=1)
    




# ax = plt.gca()
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)


# plt.xlabel("training size", fontsize=18)
# plt.ylabel("accuracy %", fontsize=18)
# plt.ylim(0,105)
# plt.xlim(0,10500)
# ax.tick_params(axis='both', which='major', labelsize=18)
# handles, labels = plt.gca().get_legend_handles_labels()
# order = [1,0]  # Desired order (indices of datasets)
# plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order],loc='upper left', bbox_to_anchor=(1, 1), fontsize=18)
# plt.subplots_adjust(right=0.7)
# # plt.margins(x=0.1, y=0.1)
# plt.savefig('rev_matching_accuracy_vs_train_size.pdf',format='pdf', bbox_inches='tight')