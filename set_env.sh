xhost +local:root

docker start droplab
docker exec -it droplab bash

docker exec -it droplab bash -c "cd ~/drop_lab_research && exec bash"