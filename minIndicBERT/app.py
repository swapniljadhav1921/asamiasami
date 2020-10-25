from flask import Flask, render_template, request, jsonify
import sys, re, traceback, gc, json
from datetime import datetime
from fairseq.models.roberta import RobertaModel
import emoji
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
import sentencepiece as spm


### App Initialization
app = Flask(__name__)


### Input Params
model_path = sys.argv[3].strip()
checkpoint_file = sys.argv[4].strip()
data_name_or_path = sys.argv[5].strip()
bpe = sys.argv[6].strip()
junk_clusters_path = sys.argv[7].strip()
clust_vect_path = sys.argv[8].strip()
clust_json_path = sys.argv[9].strip()


### RoBERTa model loading
roberta = RobertaModel.from_pretrained("%s"%model_path, checkpoint_file="%s"%checkpoint_file, data_name_or_path="%s"%data_name_or_path, bpe="%s"%bpe)
roberta.eval()
roberta.cuda()

sp = spm.SentencePieceProcessor()
sp.load("./sentencepiece.bpe.model")

### Language Supported
lang_support_list = ['hindi', 'english', 'marathi', 'tamil', 'telugu', 'kannada', 'bangla', 'gujarati', 'malayalam', 'oriya', 'punjabi', 'nepali', 'urdu']

### Clean Text Function - Basic
def clean_text(text):
    text = emoji.demojize(str(text))
    text = re.sub("\s+", " ", text.lower().strip())
    text = " ".join(text.split(' ')[:512])
    return text

###
### Read the list of junk, useless clusters
###
junk_clusters = set()
fr = open(junk_clusters_path, 'r')
for line in fr:
    junk_clusters.add(line.strip())
fr.close()
print("Junk Clusters Loaded : ", len(junk_clusters))
print(junk_clusters, "\n")

### Load Cluster Centroids Data
clust_info_df = pd.read_csv(clust_vect_path, sep="\t")
clust_centroid_dict = {}
for rowid, row in clust_info_df.iterrows():
    clust_key = row['clust_key'].strip()
    clust_centroid = [float(x) for x in row['clust_centroid'].strip().split()]
    clust_centroid_dict[clust_key] = np.array(clust_centroid)
print("len(clust_centroid_dict) : ", len(clust_centroid_dict))
del clust_info_df

### Load Cluster Hierarchy Json Data
clust_tree_data = open(clust_json_path, 'r').read()
clust_tree_dict = json.loads(clust_tree_data)
print("clust_tree_dict.keys : ", clust_tree_dict.keys())
del clust_tree_data


def get_descendant_nodes(root_node, tree_dict):
    subtree_nodes = set()
    node_stack = [root_node]
    leaf_nodes = set()
    while len(node_stack)>0:
        curr_parent_node = node_stack.pop()
        subtree_nodes.add(curr_parent_node)
        child_dict = tree_dict["parent_child_map"].get(curr_parent_node, {})
        if len(child_dict)==0:
            leaf_nodes.add(curr_parent_node)
        else:
            for child_node in child_dict.keys():
                subtree_nodes.add(child_node)
                node_stack.append(child_node)
    subtree_nodes = [tree_dict["hierarchyId_clustId_map"].get(x,x) for x in subtree_nodes]
    leaf_nodes = [tree_dict["hierarchyId_clustId_map"].get(x,x) for x in leaf_nodes]
    return subtree_nodes


def get_ancestor_nodes(child_node, tree_dict):
    ancestor_nodes = []
    child_node = tree_dict["clustId_hierarchyId_map"].get(child_node, child_node)
    curr_ancestor = tree_dict["child_parent_map"].get(child_node, None)
    while curr_ancestor is not None:
        ancestor_nodes.append(str(curr_ancestor).strip())
        curr_ancestor = tree_dict["child_parent_map"].get(curr_ancestor, None)
    return ancestor_nodes


from numba import jit

@jit(nopython=True)
def cosine_dist(u:np.ndarray, v:np.ndarray):
    uv = 0
    uu = 0
    vv = 0
    for i in range(u.shape[0]):
        uv += u[i]*v[i]
        uu += u[i]*u[i]
        vv += v[i]*v[i]
    cos_theta = 1
    if uu!=0 and vv!=0:
        cos_theta = uv/np.sqrt(uu*vv)
    return 1.0-cos_theta


###
### get roberta vector and cluster matched
###
@app.route('/get_roberta_vector_cluster', methods=['POST'])
def get_roberta_vector_cluster():
    request_info = request.json
    print("INFO :: %s :: Request Started" % datetime.now())
    start = datetime.now()
    print("INFO :: %s :: Request Started" % datetime.now())
    out_json = {"api_response" : []}
    for tmap in [request_info]:
        responce_json = {}
        try:
            ogtext, ogid = "", ""
            try:
                ogtext = tmap['text']
                ogid = tmap['id']
                ogtopn = tmap['topN']
                text = clean_text(ogtext)
                responce_json = {"id":str(ogid)}
            except:
                traceback.print_exc()
                print("ERROR :: %s :: Error in input json" % datetime.now())
                ogtext = tmap.get('text', "")
                ogid = tmap.get('id', "")
                ogtopn = 1
                responce_json = {"id":str(ogid), "text":ogtext, "topN":ogtopn}
                responce_json["error_msg"] = "Error in input json ... id or text or topN"
                out_json['api_response'].append(responce_json)
                continue
            print("INFO :: %s :: Json Received" % datetime.now())

            try:
                ### RoBERTa Vector Generation Process ###
                tokens = roberta.encode(text)
                last_layer_features = roberta.extract_features(tokens[:511]).detach().cpu().numpy()
                avg_feature = np.mean(last_layer_features[0], axis=0)
                avg_feature = avg_feature / np.linalg.norm(avg_feature)
                avg_feature = avg_feature.tolist()
                avg_feature_str = " ".join([str(x) for x in avg_feature]).strip()
                responce_json['indiebert_vector'] = avg_feature_str
                responce_json['indiebert_tokens'] = " ".join([str(x) for x in sp.encode_as_pieces(text.strip().lower())])
            except:
                traceback.print_exc()
                print("ERROR :: %s :: Error in RoBERTa Vector Generation process" % datetime.now())
                responce_json["error_msg"] = "Error in RoBERTa Vector Generation process"
                out_json['api_response'].append(responce_json)
                continue
            print("INFO :: %s :: Roberta Vector Created" % datetime.now())

            try:
                top_n = ogtopn
                ### Cluster Matching
                cosine_dist_dict = {}
                avg_feature = np.array(avg_feature)
                for clust_key, clust_centroid in clust_centroid_dict.items():
                    if clust_key not in junk_clusters:
                        cdist = cosine_dist(avg_feature, clust_centroid)
                        cosine_dist_dict[clust_key] = (1.0-cdist/2.0)
                cosine_dist_tuple = [(k, cosine_dist_dict[k]) for k in sorted(cosine_dist_dict, key=cosine_dist_dict.get, reverse=True)]
                clust_info_list = []
                num = 1
                for c, d in cosine_dist_tuple[:top_n]:
                    tmp = {}
                    tmp["topic"] = str(c).strip()
                    tmp["match_num"] = num
                    num += 1
                    tmp["topic_score"] = d
                    if tmp["topic_score"]>1: tmp["topic_score"] = 1.0
                    elif tmp["topic_score"]<0: tmp["topic_score"] = 0.0
                    tmp["topic_subtree"] = get_ancestor_nodes(c, clust_tree_dict)
                    clust_info_list.append(tmp)
                responce_json['topic_info'] = clust_info_list
            except Exception as e:
                traceback.print_exc()
                print("ERROR :: %s :: Error in Topic Match process" % datetime.now())
                responce_json["error_msg"] = "Error in Topic Match process"
                out_json['api_response'].append(responce_json)
                continue
            print("INFO :: %s :: Clusters Matched" % datetime.now())
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

