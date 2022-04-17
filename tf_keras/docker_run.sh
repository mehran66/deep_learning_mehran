docker build . --network=host -t deep_learning_tf_docker

docker run -it --rm --gpus all -v D:\deep_learning_mehran\data\:/tmp
--name tensorflow2-container --network=host deep_learning_tf_docker


docker exec -it tensorflow2-container bash -c "python3 /tmp/semantic_segmentation/main.py --step=preprocess > /tmp/semantic_segmentation/logs/preprocess.log"
