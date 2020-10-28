from utils import get_translation, Generator, sp
import sentencepiece as spm

data_token_path = "./en_hi_t2t_v3/token_data/"
model_checkpoint = "./en_hi_t2t_v3/checkpoints/checkpoint9.pt"
sp_path = "./sentencepiece.bpe.model"
srclang = "en" ### hi for hi2en

gen = Generator(data_token_path, model_checkpoint)
print("Model Checkpoint Load Complete")
sp.load(sp_path)
print("Tokenizer Load Complete\n")

while True:
    text = input("English Text : ")
    outtext = get_translation(gen, sp, text, srclang)
    print("Hindi Text : %s\n" % outtext)

