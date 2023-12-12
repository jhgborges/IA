from difflib import SequenceMatcher

# Função para calcular a similaridade entre dois códigos
def calcular_similaridade_codigo(codigo1, codigo2):
    similaridade = SequenceMatcher(None, codigo1, codigo2).ratio()
    return similaridade

# Ler códigos a partir de arquivos
with open('a1.c', 'r') as arquivo_codigo1:
    codigo1 = arquivo_codigo1.read()

with open('a2.c', 'r') as arquivo_codigo2:
    codigo2 = arquivo_codigo2.read()

# Calcular a similaridade
similaridade = calcular_similaridade_codigo(codigo1, codigo2)

# Imprimir resultado
print(f'Similaridade entre os códigos: {similaridade}')
