import argparse
from fever_io import read_jsonl, save_jsonl
from tqdm import tqdm
from jack import readers
from jack.io.load import loaders

# make everything deterministic
import random
import numpy as np
import sys
np.random.seed(1)
random.seed(1)


def reader_forward(reader, inputs):
    batch = reader.input_module(inputs)
    output_module_input = reader.model_module(batch, reader.output_module.input_ports)
    answers = reader.output_module.forward(inputs, {p: output_module_input[p] for p in reader.output_module.input_ports})
    return answers

def predict_preprocessed(reader, all_settings, batch_size):
    print("Predict preprocessed instances")
    preds_list = list()
    for pointer in tqdm(range(0, len(all_settings), batch_size)):
        batch_settings = [q for (q, a) in all_settings[pointer: pointer + batch_size]]
        t = reader_forward(reader, batch_settings)
        preds_list.extend(t)
    return preds_list  

def snli_format(captionID, pairID, label, sentence1, sentence2, predicted_label):
    return {
        "captionID": captionID,
        "pairID": pairID,
        "gold_label": label,
        "sentence1": sentence1,
        "sentence2": sentence2,
        "predicted_label": predicted_label
    }

def save_predictions_preprocessed(instances, preds_list, path):
    
    print("prepare saved predictions")
    print('instances', len(instances), 'preds_list', len(preds_list))
    assert len(instances) == len(preds_list)
    
    store = []
    for instance, pred in tqdm(zip(instances, preds_list)):
        pred.sort(key=lambda x: x.score, reverse=True)
        predicted_label = pred[0].text
        store.append(snli_format(
            instance["captionID"],
            instance["pairID"],
            instance["gold_label"],
            instance["sentence1"],
            instance["sentence2"],
            predicted_label,
        ))
    
    save_jsonl(store, path)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser("read claim/evidence and output verdict")
    parser.add_argument("in_file",
        help="input file path for rte (e.g., dev.sentences.p5.s5.jsonl)")
    parser.add_argument("save_preds", help="specify file name to save prediction")
    parser.add_argument(
        "--saved_reader", help="path to saved reader directory")
    parser.add_argument("--batch_size", type=int, default=512, help="batch size for inference")
    parser.add_argument("--preprocessed_in_file", default=None, type=str,
                        help="path for preprocessed input data, if set, in_file is ignored")
    args = parser.parse_args()
    print(args, file=sys.stderr)

    print("loading reader from file:", args.saved_reader)
    dam_reader = readers.reader_from_file(args.saved_reader, dropout=0.0)

    instances = read_jsonl(args.preprocessed_in_file)
    instances = [instance for instance in instances 
        if instance['gold_label'] in ['contradiction', 'entailment', 'neutral']]
    
    print('Use preprocessed file', file=sys.stderr)
    loader = 'snli'
    all_settings = loaders[loader](args.preprocessed_in_file)
    preds_list = predict_preprocessed(dam_reader, all_settings, args.batch_size)

    save_predictions_preprocessed(instances, preds_list, path=args.save_preds)
