# Chaining thoughts and LLMs to learn DNA structural biophysics

This repo contains all the code/data used for [Chaining thoughts and LLMs to learn DNA structural biophysics](http://arxiv.org/abs/2403.01332) by Tyler Ross and Ashwin Gopinath

#### Setup

To get started:

1. Clone this repo and move into the directory:
```bash
git https://github.com/TDRoss/DNA-LLM
```

2. Install the module dependencies into your environment:
```bash
pip install -r requirements.txt
```

3. Set `OPENAI_API_KEY` environment variable to your OpenAI API key:
```bash
export OPENAI_API_KEY=<your key>
```


#### Generating training and validation sets

Sequence analysis data sets are generated by running `generate_training_sequences.py` and sequence design data sets are generated by running `generate_training_structures`. Resulting data sets are saved as json files in `/training_data`.

#### Fine-tuning and validation
Fine-tuning is performed in `fine-tune.py`, at the bottom of the file is where the script setup takes place. There are four types of experiments, and for each experiment there are a set of possible conditions. They are as follows:

- reverse_complement
	- naive
	- CoT
- secondary_structure
	- naive
	- seq2CoT (we refer to this as CoT in the manuscript)
	- rev2CoT (prints out reverse complement before performing CoT)
	- +rev_comp+CoT (For the pipeline where the user input now has the reverse complement of sequence 2)
- minimum_free_energy
    - naive
    - rev2CoT
    - +rev_comp+CoT
    - +rev_comp+dotpar
- sequence_design
    - naive
    - CoTseq2 (CoT & rev. comp. in the manuscript)
    - CoTrev2+rev_comp (for pipeline approaches)

There is also the additional variable of training size. Once fine tuning is complete a json file is crated in `/model_ids` where each entry is a list containing the training size used and the OpenAI model ID.

Once the fine-tuning is complete the validation is performed with `performance_test.py`. This uses a similar input scheme as with the fine-tuning but with some additional terms. `max_tries` sets the number of retries the model in which the model is determined to have failed for the given input (for which model answers are set to "2"). For pipelines of experts, the `condition` name is appended with `_expert_tries_{n}` where `{n}` is the maximum number of retries that an expert gets.

The names of the conditions are the same as those used in fine-tuning except for where pipelines of experts are used. Pipeline conditions are as follows:

- secondary_structure
	- +rev_comp_expert+CoT (expert reverse complement)
	- +rev_comp+base_compare+CoT (ground truth reverse complement)
- minimum_free_energy
	- +rev_comp_expert+CoT (expert reverse complement)
	- +rev_comp+CoT (ground truth reverse complement)
- sequence_design
	- +CoTrev2+rev_comp_expert (expert reverse complement)
	- +CoTrev2+rev_comp (ground truth reverse complement)
	- +CoTrev2+rev_comp_expert+error_checking_expert+ (expert reverse complement with expert error check)
	- +CoTrev2+rev_comp_expert+error_checking+ (expert reverse complement with ground truth error check)

The validation results are saved as json files to `/test_results`. For the reverse complement experiment, `rev2` is the ground truth reverse complement and `model` is the model's predicted reverse complement. In the secondary structure experiment, `structure` is the secondary structure ground truth and `model_structure` is the model's predicted secondary structure. In the cases where the reverse complement expert is used in the pipeline, that output is saved as `model_rev2`. In the minimum free energy experiments, `MFE` is the ground truth minimum free energy in kcal/mol and `model_MFE` is the predicted minimum free energy. Sequence design saves the input structure as `structure`, the model generated sequences as `model_seq1` and `model_seq2`, and the ground truth structure that they form as `model_structure`. When expert error checking is used, the predicted structure is saved as `expert_dotpar`.

#### Analyzing results

Performance of the models are evaluated by running `analyze_results.py`, where the values are printed to the terminal. Learning curve plots are generated by running `plot_learning_curves.py` where the resulting plots are saved as PDFs in the project directory.

#### Cite

```bibtex
@misc{shinn2023reflexion,
      title={Chaining thoughts and LLMs to learn DNA structural biophysics}, 
      author={Tyler D. Ross and Ashwin Gopinath},
      year={2024},
      eprint={2403.01332},
      archivePrefix={arXiv},
}
```


