#!/bin/bash
echo "------------- COPIAR (ctrl-c) somente a partir da linha abaixo: -----------"

echo "Executando 10 vezes com Q: 128 pontos, 300 dimensoes e P: 400 mil, 300 dimensoes (Local, 1 nucleo):"    
for vez in $(seq 1 10)  # 10 vezes
do
    mpirun -np 1 ./knn 128 400000 300 128 2> /dev/null | grep -oP '(?<=total_time_in_seconds: )[^ ]*'
done
echo "Executando 10 vezes com Q: 128 pontos, 300 dimensoes e P: 400 mil, 300 dimensoes (Local, 4 nucleos):"  
for vez in $(seq 1 10)  # 10 vezes
do
    mpirun -np 4 ./knn 128 400000 300 128 2> /dev/null | grep -oP '(?<=total_time_in_seconds: )[^ ]*'
done
echo "Executando 10 vezes com Q: 128 pontos, 300 dimensoes e P: 400 mil, 300 dimensoes (cluster, 4 pc's):"  
for vez in $(seq 1 10)  # 10 vezes
do
    mpirun -np 4 --hostfile hostfile.txt ./knn 128 400000 300 128 2> /dev/null | grep -oP '(?<=total_time_in_seconds: )[^ ]*'
done

#