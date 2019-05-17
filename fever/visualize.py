from jack import readers
from jack.core import QASetting

saved_reader='/scratch/fz758/ucl/fever/results/depsa_elmo_10k_n5/reader'

reader = readers.reader_from_file(saved_reader, dropout=0.0)
reader.model_module.prediction_module.set_visualize('/scratch/fz758/ucl/fever/results/depsa_elmo_10k_n5/dep_attn.jsonl')


s = 'Munich is the capital and largest city of the German state of Bavaria.'
s_tokenized = ['Munich', 'is', 'the', 'capital', 'and', 'largest', 'city', 'of', 'the', 'German', 'state', 'of', 'Bavaria', '.']
s_length = [14]
s_dep_i = [3, 3, 3, 6, 6, 3, 10, 10, 10, 6, 12, 10, 3]
s_dep_j = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
s_dep_type = [6, 19, 7, 17, 12, 14, 5, 7, 12, 16, 5, 16, 4]

q = 'Munich is the capital of Germany.'
q_tokenized = ['Munich', 'is', 'the', 'capital', 'of', 'Germany', '.']
q_length = [7]
q_dep_i = [3, 3, 3, 5, 3, 3]
q_dep_j = [0, 1, 2, 4, 5, 6]
q_dep_type = [6, 19, 7, 5, 16, 4]

qa_setting = QASetting(
    q,
    [s],
    1,
    None,
    None,
    None,
    q_tokenized = q_tokenized,
    s_tokenized = s_tokenized,
    q_dep_i = q_dep_i,
    q_dep_j = q_dep_j,
    q_dep_type = q_dep_type,
    s_dep_i = s_dep_i,
    s_dep_j = s_dep_j,
    s_dep_type = s_dep_type
)

batch = reader.input_module([qa_setting])
output_module_input = reader.model_module(batch)
answers = reader.output_module.forward([qa_setting], 
    {p: output_module_input[p] for p in reader.output_module.input_ports})
