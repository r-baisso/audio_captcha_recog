# Projeto Reconhecimento de Audios
### Materia de Mineracao de Dados UFABC 

 * Para este projeto e necessario que as pastas com os dados de treino e validacao estejam nas pastas 'TREINAMENTO/' e 'VALIDACAO/', respectivamente, dentro da pasta principal do projeto.
 * Os dados devem estar formato .wav e duracao maxima de ~ 8s.

 ### Bibliotecas utilizadas
* librosa
* glob
* random
* pandas
* numpy 
* matplotlib
* re
* sklearn
* scipy (not used)
* math (not used)

### Como Executar
* Apos configurar as pastas com os dados para executar o projeto, basta  realizar o comando:
```
     python project.py
```
* Tomar o devido cuidado para nao executar utilizando python 2.

* Os testes foram realizados tendo o Python 3.6 como versão default do sistema.

### Resultados iniciais

* Para o classificador Random Forest obtivemos uma taxa de acerto de aproximadamente 74% (Nath: 73,88% com shuffle e 73,97% sem shuffle), para 500 estimators e max_depth = 50.

* Para o classificador SVC (kernel linear) obtivemos uma taxa de acerto de aproximadamente 60% (Nath: 59,88% aplicando shuffle nos files e 59,93% sem aplicar).

* Para o classificador KNN (3 neighbours) obtivemos uma taxa de acerto de aproximadamente 57% (Nath: 57,4%).

* Em ambos os classificadores a taxa de acerto dos caracteres n, m, d e b foi mais baixa que a dos demais.
