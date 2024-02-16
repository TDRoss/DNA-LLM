import json
import nupack as nup


def structure_from_strands(strand1, strand2, nupackmodel):
    A = nup.Strand(strand1, name='A')
    B = nup.Strand(strand2, name='B')
    c1 = nup.Complex([A,B]) 
    complex_set = nup.ComplexSet(strands={A: 1e-8, B: 1e-8}, complexes=nup.SetSpec(max_size=0, include=[c1]))
    complex_analysis = nup.complex_analysis(complex_set, compute=['mfe','pfunc', 'pairs'], model=nupackmodel)
    complex_vals = complex_analysis[c1]
    structure = str(complex_vals.mfe[0].structure)
    return structure    

nupackmodel = nup.Model(material='DNA',celsius=20)

responses = []
with open("test_results/sequence_design_CoTrev2+rev_comp_expert+error_checking_expert+_expert_tries_20_max_tries_20_test_size_10000.json",'r') as f:
    for line in f:
        responses.append(json.loads(line))

new_version = []
for entry in responses:
    if entry["model_seq2"] != "2":
        actual_model_structure = structure_from_strands(entry["model_seq1"],entry["model_seq2"],nupackmodel)
    else:
         actual_model_structure = "2"
    new_version.append({
        "structure":entry["structure"],
        "model_structure":actual_model_structure,
        "expert_structure":entry["model_structure"],
        "model_seq1":entry["model_seq1"],
        "model_seq2":entry["model_seq2"]
    })

    with open("test_results/sequence_design_CoTrev2+rev_comp_expert+error_checking_expert+_expert_tries_20_max_tries_20_test_size_10000_new.json", 'w') as f:
        for item in new_version:
            f.write(json.dumps(item) + '\n')
