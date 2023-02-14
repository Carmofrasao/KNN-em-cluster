all: knn

knn: knn.c chrono.c
	mpic++ knn.c -O3 -g -o knn