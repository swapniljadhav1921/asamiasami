from flask import Flask, render_template, request, jsonify
import sys, re, traceback, gc, json
from datetime import datetime
from fairseq.models.roberta import RobertaModel
import emoji
import numpy as np


### App Initialization
app = Flask(__name__)

model_path = sys.argv[3].strip()
checkpoint_file = sys.argv[4].strip()
data_name_or_path = sys.argv[5].strip()
bpe = sys.argv[6].strip()
roberta = RobertaModel.from_pretrained(model_path, checkpoint_file=checkpoint_file, data_name_or_path=data_name_or_path, bpe=bpe)
roberta.eval()
roberta.cuda()


nsfw_list = set()
fr = open(sys.argv[7].strip(), 'r')
for line in fr:
    text = line.strip()
    if len(text)>0 and text != "" and text is not None:
        nsfw_list.add(text)
fr.close()
print("nsfw stopwords count : ", len(nsfw_list))


def get_default_responce(text, responce_json):
    words = text.split()
    phrase_set = set()
    irange = np.min([6, len(words)])
    for i in range(1,irange, 1):
        for j in range(len(words)):
            itext = " ".join(words[j:j+i]).strip()
            if itext is not None and len(itext)>1:
                phrase_set.add(itext)
    nsfw_count = 0
    nw_list = []
    for nw in phrase_set:
        if nw in nsfw_list:
            nsfw_count += 1
            nw_list.append(nw)
    responce_json["nsfw_text"] = nw_list
    return responce_json



label_fn = lambda label: roberta.task.label_dictionary.string([label + roberta.task.label_dictionary.nspecial])
lang_arr = ['sfw', 'nsfw']

def clean_text(text):
    text = emoji.demojize(str(text))
    text = text.lower().replace("  ", " ").replace("  ", " ").replace("  ", " ").strip()
    chstr = "[!@#$%^&*()[]{};:,./<>?\|`~-=_+]"
    for ch in chstr:
        text = text.replace(ch, " ")
    text = text.replace("  ", " ").replace("  ", " ").replace("  ", " ").strip()
    text = re.sub("\s+", " ", text)
    return text


###
### roberta NSFW detection
###
@app.route('/get_nsfw_scores', methods=['POST'])
def get_nsfw_scores():
    start = datetime.now()
    print("INFO :: %s :: Request Started" % datetime.now())
    request_info = request.json
    out_json = {"api_response" : []}
    for tmap in [request_info]:
        responce_json = {}
        try:
            try:
                ogtext = tmap['text']
                ogid = tmap['id']
                text = clean_text(ogtext)
                responce_json = {"id":str(ogid)}
            except:
                traceback.print_exc()
                print("ERROR :: %s :: Error in input json" % datetime.now())
                ogtext = tmap.get('text', "")
                ogid = tmap.get('id', "")
                responce_json = {"id":str(ogid), "text":ogtext}
                responce_json["error_msg"] = "Error in input json id or text"
                out_json['api_response'].append(responce_json)
                continue
            print("INFO :: %s :: Json Received" % datetime.now())
            try:
                ### RoBERTa NSFW Classification Process ###
                tokens = roberta.encode(text)
                print("INFO :: %s :: RoBERTa encoding done" % datetime.now())
                pred_arr = roberta.predict('nsfw_detect', tokens[:511]).detach().cpu().numpy().tolist()[0] ### GPU
                print("INFO :: %s :: RoBERTa prediction done" % datetime.now())
                pred_min = np.min(pred_arr)
                if pred_min < 0:
                    pred_min = pred_min * -1.01
                    pred_arr = [x+pred_min for x in pred_arr]
                pred_sum = np.sum(pred_arr)
                pred_arr = [float("{:.4f}".format(x*1.0/pred_sum)) for x in pred_arr]
                possible_classes = dict(zip(lang_arr, pred_arr))
                responce_json['score_map'] = possible_classes
                ### Possible nsfw text
                responce_json = get_default_responce(text[:511], responce_json)
            except:
                traceback.print_exc()
                print("ERROR :: %s :: Error in RoBERTa NSFW Classification process" % datetime.now())
                responce_json["error_msg"] = "Error in RoBERTa NSFW Classification process"
                out_json['api_response'].append(responce_json)
                continue
        except Exception as e:
            print("ERROR :: %s :: %s - %s" % (datetime.now(), traceback.format_exc(), e))
            responce_json["error_msg"] = "%s : %s" % (traceback.format_exc(), e)
        out_json['api_response'].append(responce_json)
    end = datetime.now()
    print("INFO :: %s :: Processing Done" % datetime.now())
    out_json['api_time'] = "%s seconds" % (end-start).total_seconds()
    print("INFO :: %s :: API Time - %s" % (datetime.now(), (end-start).total_seconds()))
    return jsonify(out_json)

@app.route('/healthcheck', methods=['POST', 'GET'])
def healthcheck():
    gc.collect()
    return jsonify({"message":"success"})


if __name__ == '__main__':
    app_host = sys.argv[1].strip()
    app_port = sys.argv[2].strip()
    app.run(use_reloader = False, debug = True, host = app_host, port = app_port)
