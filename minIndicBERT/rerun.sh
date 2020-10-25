port=$1

sudo kill -9 `sudo lsof -t -i:$port`

cd /home/ubuntu/demo_apis/indiebert/

nohup python -u app.py 0.0.0.0 $port /home/ubuntu/demo_apis/indiebert/ checkpoint_best.pt ./token_data sentencepiece junk_clusters.txt clust_centroids.txt clust_hierarchy.json > nohup.out &

echo "minIndicBERT api started/restarted"
