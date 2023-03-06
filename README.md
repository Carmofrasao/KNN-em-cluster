# KNN em cluster

Trabalho feito para a matéria de programação paralela, no curso de Ciencia da Computação, da Universidade Federal do Paraná.

Autor:
Anderson Frasão

Programa para calcular os k mais proximos de um ponto.

Para ultilizar o programa, execute o comando make no terminal (dentro da pasta em que o código esta) e execute um desses comandos:

> ./knn

ou

> mpirun -np < np > ./knn < nq > < n > < D > < k >

sendo: 

* np: numero de processos

* n: numero de pontos de uma matriz
  
* nq: numero de pontos de outra martriz
  
* D: numero de dimensões de ambas matrizes
  
* k: numero de vizinhos calculados para cada ponto
