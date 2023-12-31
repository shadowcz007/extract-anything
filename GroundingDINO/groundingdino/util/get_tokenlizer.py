from transformers import AutoTokenizer, BertModel, BertTokenizer, RobertaModel, RobertaTokenizerFast

import os
current_path = os.path.abspath(os.path.dirname(__file__))
mp=os.path.abspath(os.path.join(current_path,'../../../checkpoints/bert-base-uncased'))


def get_tokenlizer(text_encoder_type):
    if not isinstance(text_encoder_type, str):
        # print("text_encoder_type is not a str")
        if hasattr(text_encoder_type, "text_encoder_type"):
            text_encoder_type = text_encoder_type.text_encoder_type
        elif text_encoder_type.get("text_encoder_type", False):
            text_encoder_type = text_encoder_type.get("text_encoder_type")
        else:
            raise ValueError(
                "Unknown type of text_encoder_type: {}".format(type(text_encoder_type))
            )
    print("final text_encoder_type: {}".format(text_encoder_type))

    tokenizer = AutoTokenizer.from_pretrained(mp)
    return tokenizer


def get_pretrained_language_model(text_encoder_type):
    
    if text_encoder_type == "bert-base-uncased":
        print("bert-base-uncased:::::",mp)
        return BertModel.from_pretrained(mp)
    if text_encoder_type == "roberta-base":
        return RobertaModel.from_pretrained(text_encoder_type)
    raise ValueError("Unknown text_encoder_type {}".format(text_encoder_type))
