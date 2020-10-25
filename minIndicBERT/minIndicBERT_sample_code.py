from fairseq.models.roberta import RobertaModel
import numpy as np
import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.load("./sentencepiece.bpe.model")

roberta = RobertaModel.from_pretrained("/data/public_share/asamiasami/minIndicBERT", checkpoint_file="checkpoint_best.pt", data_name_or_path="./token_data", bpe="sentencepiece")
roberta.eval()
roberta.cuda()

text = """चिराग पासवान ने ट्वीट करके दावा किया कि चुनाव के बाद बिहार में नीतीश मुक्त सरकार बनेगी. उन्होंने कहा कि जनता नीतीश कुमार के राज से परेशान हो चुकी है. बीते 15 सालों में बिहार को बेरोजगारी, गरीबी, अशिक्षा के अलावा कुछ नहीं मिला."""
tokens = roberta.encode(text)
last_layer_features = roberta.extract_features(tokens[:511]).detach().cpu().numpy()
avg_feature = np.mean(last_layer_features[0], axis=0)
avg_feature = avg_feature / np.linalg.norm(avg_feature)
print("minIndicBERT Vector => ", avg_feature)
print("\ntokens => ", " ".join([str(x) for x in sp.encode_as_pieces(text.strip().lower())]))
