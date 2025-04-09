
# RVV-SIMD: Biblioteca SIMD Otimizada para RISC-V Vector

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/<your_username>/rvv-simd/actions) <!-- Substitua pelo link real do CI -->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/)
[![pybind11](https://img.shields.io/badge/bindings-pybind11-orange.svg)](https://github.com/pybind/pybind11)

Uma biblioteca SIMD (Single Instruction, Multiple Data) de alto desempenho otimizada para a extensão vetorial de RISC-V (RVV), com bindings Python, projetada para aplicações de machine learning e processamento de dados.

## Sumário

1.  [Introdução](#introdução)
2.  [Arquitetura](#arquitetura)
3.  [Instalação](#instalação)
4.  [Uso da Biblioteca em C++](#uso-da-biblioteca-em-c)
5.  [Uso da Biblioteca em Python](#uso-da-biblioteca-em-python)
6.  [Benchmarks Comparativos](#benchmarks-comparativos)
7.  [Aplicações em Machine Learning](#aplicações-em-machine-learning)
8.  [Otimizações para RVV](#otimizações-para-rvv)
9.  [Alternativas e Comparação](#alternativas-e-comparação)
10. [Estado Atual do Suporte Python (Considerações)](#estado-atual-do-suporte-python-considerações)
11. [Estrutura do Projeto](#estrutura-do-projeto)
12. [Contribuições](#contribuições)
13. [Licença](#licença)
14. [Agradecimentos](#agradecimentos)

## Introdução

A biblioteca RVV-SIMD é uma implementação de operações SIMD otimizadas para a extensão vetorial de RISC-V (RVV). Esta biblioteca visa preencher uma lacuna importante no ecossistema RISC-V, especialmente em aplicações de machine learning (ML) e outras áreas que se beneficiam de processamento paralelo intensivo.

### Motivação

A escassez de bibliotecas otimizadas para RVV tem sido um obstáculo para a adoção mais ampla do RISC-V em aplicações de ML. Esta biblioteca foi desenvolvida para:

1.  **Explorar o potencial da extensão vetorial de RISC-V**: Utilizando instruções RVV para acelerar operações paralelas.
2.  **Democratizar o acesso à computação vetorial em RISC-V**: Através de bindings Python que facilitam a integração com frameworks populares.
3.  **Fornecer benchmarks comparativos**: Comparando o desempenho com arquiteturas x86 (usando AVX) e ARM (usando NEON).
4.  **Suportar aplicações de ML**: Implementando operações comuns em redes neurais e outros algoritmos de ML.

### Características Principais

*   **Operações vetoriais otimizadas**: Implementações eficientes de operações básicas em vetores.
*   **Operações matriciais**: Suporte a operações em matrizes, incluindo multiplicação e transposição.
*   **Operações de ML**: Implementações de convolução, pooling, batch normalization e outras operações comuns em ML.
*   **Bindings Python**: Interface Python completa, compatível com NumPy.
*   **Benchmarks comparativos**: Ferramentas para comparar o desempenho com x86 (AVX) e ARM (NEON).
*   **Implementações de fallback**: Código escalar para plataformas sem suporte a RVV, garantindo portabilidade.

## Arquitetura

A biblioteca RVV-SIMD é estruturada em camadas para fornecer tanto operações de baixo nível otimizadas quanto interfaces de alto nível para aplicações de ML e processamento de dados.

### Componentes Principais

#### Biblioteca Core (C++)

A biblioteca core é implementada em C++ e consiste nos seguintes componentes:

1.  **Operações Vetoriais**:
    *   Operações aritméticas básicas (adição, subtração, multiplicação, divisão)
    *   Produto escalar (dot product)
    *   Escalonamento de vetores
    *   Normalização de vetores
    *   Funções matemáticas (exp, log, sigmoid, tanh, ReLU)
2.  **Operações Matriciais**:
    *   Operações aritméticas em matrizes (adição, subtração, multiplicação elemento a elemento)
    *   Multiplicação de matrizes
    *   Transposição de matrizes
    *   Escalonamento de matrizes
    *   Normas de matrizes
3.  **Operações de Machine Learning**:
    *   Operações de convolução para CNNs
    *   Operações de pooling (max, average)
    *   Batch normalization
    *   Funções de ativação (softmax)
    *   Funções de perda (cross-entropy)
    *   Cálculo de gradientes

#### Bindings Python

Os bindings Python, implementados com `pybind11`, fornecem uma interface de alto nível para a biblioteca core, tornando-a acessível para usuários Python e integrando-a com o ecossistema de ciência de dados do Python:

*   Interface compatível com NumPy (aceita e retorna arrays NumPy)
*   Suporte para arrays multidimensionais
*   Integração facilitada com frameworks de ML do Python (PyTorch, TensorFlow, etc.)

### Diagrama de Arquitetura

```
+---------------------+
|    Camada Python    |
| (Compatível NumPy)  |
+----------+----------+
           |
           v
+----------+----------+
|   Bindings Python   |
|     (pybind11)      |
+----------+----------+
           |
           v
+---------------------+
| Biblioteca Core C++ |
+---------------------+
|  Operações Vetoriais|
|  Operações Matriciais|
|  Operações de ML    |
+----------+----------+
           |
           v
+---------------------+    +------------------------+
| Instruções Vetoriais|--->| Implementação Fallback |
|     RISC-V (RVV)    |    |    (Escalar / C++)     |
+---------------------+    +------------------------+
 (Se suportado)             (Se RVV não suportado)
```

### Detalhes de Implementação

#### Extensão Vetorial RISC-V (RVV)

A biblioteca utiliza intrínsecos vetoriais RISC-V (`<riscv_vector.h>`) quando compilada para hardware compatível. As otimizações aproveitam a flexibilidade da RVV, como o ajuste dinâmico do comprimento do vetor (`vl`) e o uso eficiente dos registradores vetoriais. Principais intrínsecos utilizados incluem:

*   `__riscv_vsetvl_e32m8` (e variantes): Define o comprimento do vetor para processamento.
*   `__riscv_vle32_v_f32m8`: Carrega elementos de memória para registradores vetoriais.
*   `__riscv_vse32_v_f32m8`: Armazena elementos de registradores vetoriais na memória.
*   `__riscv_vfadd_vv_f32m8`, `vfsub`, `vfmul`, `vfdiv`: Operações aritméticas vetoriais.
*   `__riscv_vfmacc_vv_f32m8`: Multiplicação-acumulação vetorial (útil em matmul, conv).
*   `__riscv_vfredusum_vs_f32m8_f32m1`: Redução de soma vetorial (útil em dot product).
*   Operações de máscara para execução condicional.

#### Implementações de Fallback

Para garantir a portabilidade e usabilidade em plataformas RISC-V sem a extensão vetorial ou em outras arquiteturas (para fins de teste/comparação), a biblioteca fornece implementações escalares puras em C++ para todas as operações. A seleção entre a implementação RVV e a de fallback é feita em tempo de compilação usando diretivas de pré-processador (`#ifdef __riscv_vector`).

## Instalação

### Pré-requisitos

*   **Toolchain RISC-V**: Compilador (GCC ou Clang) com suporte à extensão vetorial (`-march=rv64gcv` ou similar).
*   **CMake**: Versão 3.10 ou superior.
*   **Python**: Versão 3.6 ou superior (necessário apenas para os bindings Python).
*   **pybind11**: Biblioteca C++ para criar bindings Python (geralmente incluída como submódulo ou baixada pelo CMake).
*   **NumPy**: Biblioteca Python para manipulação de arrays (necessária para os exemplos e testes Python).

### Compilando a partir do Código Fonte

1.  **Clone o repositório:**
    ```bash
    git clone https://github.com/<your_username>/rvv-simd.git # Substitua pelo URL real
    cd rvv-simd
    # Opcional: Inicializar submódulos (se pybind11 for um submódulo)
    # git submodule update --init --recursive
    ```

2.  **Crie um diretório de build e configure com CMake:**
    ```bash
    mkdir build && cd build
    # Para build padrão (detectará RVV se o toolchain suportar):
    cmake ..
    # Para forçar build com RVV (requer toolchain compatível):
    # cmake .. -DRVV_SIMD_FORCE_RVV=ON
    # Para forçar build com fallback (útil para testes em x86/ARM):
    # cmake .. -DRVV_SIMD_FORCE_FALLBACK=ON
    # Para habilitar build dos bindings Python:
    # cmake .. -DRVV_SIMD_BUILD_PYTHON=ON
    ```

3.  **Compile a biblioteca:**
    ```bash
    make -j$(nproc) # Compila em paralelo
    ```

4.  **(Opcional) Instale a biblioteca:**
    ```bash
    sudo make install # Instala headers e biblioteca no sistema
    ```

### Instalando os Bindings Python

Se você habilitou a opção `DRVV_SIMD_BUILD_PYTHON=ON` no CMake e compilou:

1.  **Navegue até o diretório Python:**
    ```bash
    cd ../python # A partir do diretório 'build'
    ```

2.  **Instale o pacote Python em modo editável:**
    ```bash
    pip install -e .
    ```
    Isso cria um link para o módulo compilado no diretório `build`, permitindo que você importe `rvv_simd` em Python.

## Uso da Biblioteca em C++

### Incluindo a Biblioteca

```cpp
#include "rvv_simd.h" // Ou caminho específico, e.g., <rvv_simd/vector_ops.h>
```

### Inicializando e Verificando Suporte

```cpp
#include <iostream>
#include "rvv_simd/core.h" // Para funções de inicialização/info

int main() {
    // Inicializa a biblioteca (pode realizar verificações de hardware)
    if (!rvv_simd::initialize()) {
        std::cerr << "Falha ao inicializar a biblioteca RVV-SIMD" << std::endl;
        return 1;
    }

    // Verifica se RVV é suportado e ativo
    if (rvv_simd::is_rvv_supported()) {
        std::cout << "RVV é suportado e está sendo utilizado." << std::endl;
        std::cout << "Informações RVV: " << rvv_simd::get_rvv_info() << std::endl;
    } else {
        std::cout << "RVV não é suportado ou desabilitado. Usando implementações de fallback." << std::endl;
    }

    std::cout << "Versão RVV-SIMD: " << rvv_simd::get_version() << std::endl;

    // ... seu código aqui ...

    return 0;
}
```

### Operações Vetoriais

```cpp
#include "rvv_simd/vector_ops.h"
#include <vector>
#include <numeric>

// Cria vetores de entrada (exemplo com std::vector)
const size_t size = 1024;
std::vector<float> a(size);
std::vector<float> b(size);
std::vector<float> result(size);

// Inicializa vetores com dados (exemplo)
std::iota(a.begin(), a.end(), 1.0f);
std::iota(b.begin(), b.end(), 0.5f);

// Adição de vetores
rvv_simd::vector_add(a.data(), b.data(), size, result.data());

// Produto escalar
float dot_product = rvv_simd::vector_dot(a.data(), b.data(), size);

// Escalonamento de vetor
rvv_simd::vector_scale(a.data(), 2.5f, size, result.data());

// Normalização de vetor
rvv_simd::vector_normalize(a.data(), size, result.data());

// Funções de ativação (exemplo com ReLU)
rvv_simd::vector_relu(a.data(), size, result.data());
```

### Operações Matriciais

```cpp
#include "rvv_simd/matrix_ops.h"
#include <vector>

// Cria matrizes de entrada (layout row-major)
const size_t a_rows = 32, a_cols = 64;
const size_t b_rows = 64, b_cols = 32; // b_rows deve ser igual a a_cols
std::vector<float> a_mat(a_rows * a_cols);
std::vector<float> b_mat(b_rows * b_cols);
std::vector<float> c_mat(a_rows * b_cols);

// Inicializa matrizes com dados...

// Multiplicação de matrizes (C = A * B)
rvv_simd::matrix_mul(a_mat.data(), b_mat.data(), a_rows, a_cols, b_cols, c_mat.data());

// Transposição de matriz
std::vector<float> a_transpose(a_cols * a_rows);
rvv_simd::matrix_transpose(a_mat.data(), a_rows, a_cols, a_transpose.data());
```

### Operações de Machine Learning

```cpp
#include "rvv_simd/ml_ops.h"
#include <vector>

// Exemplo: Operação de convolução 2D
const size_t input_c = 3, input_h = 32, input_w = 32;
const size_t kernel_n = 16, kernel_h = 3, kernel_w = 3;
const size_t stride_h = 1, stride_w = 1;
const size_t padding_h = 1, padding_w = 1;

std::vector<float> input(input_c * input_h * input_w);
std::vector<float> kernel(kernel_n * input_c * kernel_h * kernel_w);
// Inicializa input e kernel...

// Calcula dimensões de saída
const size_t output_h = (input_h + 2 * padding_h - kernel_h) / stride_h + 1;
const size_t output_w = (input_w + 2 * padding_w - kernel_w) / stride_w + 1;
std::vector<float> output(kernel_n * output_h * output_w);

rvv_simd::convolution_2d(
    input.data(), kernel.data(),
    input_h, input_w, input_c,
    kernel_h, kernel_w, input_c, kernel_n, // Nota: Assumindo layout de kernel NCHW
    stride_h, stride_w,
    padding_h, padding_w,
    output.data()
);

// Exemplo: Max pooling 2D
const size_t pool_h = 2, pool_w = 2;
const size_t pool_stride_h = 2, pool_stride_w = 2;
const size_t pooled_h = (output_h - pool_h) / pool_stride_h + 1;
const size_t pooled_w = (output_w - pool_w) / pool_stride_w + 1;
std::vector<float> pooled(kernel_n * pooled_h * pooled_w);

rvv_simd::max_pooling_2d(
    output.data(),
    output_h, output_w, kernel_n,
    pool_h, pool_w,
    pool_stride_h, pool_stride_w,
    pooled.data()
);
```

## Uso da Biblioteca em Python

### Importando a Biblioteca

```python
import numpy as np
import rvv_simd as rv

# Verifica se a versão com RVV está ativa (se aplicável)
print(f"RVV-SIMD Version: {rv.get_version()}")
print(f"RVV Supported: {rv.is_rvv_supported()}")
if rv.is_rvv_supported():
    print(f"RVV Info: {rv.get_rvv_info()}")
```

### Operações Vetoriais

```python
# Cria vetores NumPy (precisão float32 é comum)
size = 1024
a = np.random.uniform(-10, 10, size).astype(np.float32)
b = np.random.uniform(-10, 10, size).astype(np.float32)

# Adição de vetores (aceita e retorna NumPy arrays)
c = rv.vector_add(a, b)
# API alternativa estilo NumPy (se implementada)
# c = rv.add(a, b)

# Produto escalar
dot_product = rv.vector_dot(a, b)
# dot_product = rv.dot(a, b)

# Escalonamento de vetor
scaled = rv.vector_scale(a, 2.5)

# Normalização de vetor
normalized = rv.vector_normalize(a)

# Funções de ativação
sigmoid_result = rv.sigmoid(a) # Nome hipotético, use o nome real da API
relu_result = rv.relu(a)       # Nome hipotético, use o nome real da API
```

### Operações Matriciais

```python
# Cria matrizes NumPy
rows, cols = 32, 32
a = np.random.uniform(-10, 10, (rows, cols)).astype(np.float32)
b = np.random.uniform(-10, 10, (rows, cols)).astype(np.float32)

# Adição de matrizes
c = rv.matrix_add(a, b)

# Multiplicação de matrizes
a_rows, a_cols, b_cols = 32, 64, 32
a_mat = np.random.uniform(-10, 10, (a_rows, a_cols)).astype(np.float32)
b_mat = np.random.uniform(-10, 10, (a_cols, b_cols)).astype(np.float32)

c_mat = rv.matrix_mul(a_mat, b_mat)
# API alternativa estilo NumPy
# c_mat = rv.matmul(a_mat, b_mat)

# Transposição de matriz
a_transpose = rv.matrix_transpose(a)
# a_transpose = rv.transpose(a)
```

### Operações de Machine Learning

```python
# Cria tensor de entrada (ex: NCHW - batch, canais, altura, largura)
# Nota: A API Python pode esperar um layout específico. Verifique a documentação.
batch_size = 1 # Exemplo com batch 1
input_channels = 3
input_height = 32
input_width = 32
input_tensor = np.random.uniform(-1, 1, (batch_size, input_channels, input_height, input_width)).astype(np.float32)

# Cria tensor de kernel (ex: NCHW - num_kernels, canais_in, altura, largura)
kernel_num = 16
kernel_height = 3
kernel_width = 3
kernel_tensor = np.random.uniform(-1, 1, (kernel_num, input_channels, kernel_height, kernel_width)).astype(np.float32)

# Operação de convolução
stride = (1, 1)
padding = (1, 1)
# A API pode ter nomes/parâmetros ligeiramente diferentes. Exemplo:
# output = rv.conv2d(input_tensor, kernel_tensor, stride=stride, padding=padding)
output = rv.convolution_2d(input_tensor[0], kernel_tensor, stride[0], stride[1], padding[0], padding[1]) # Exemplo adaptado da API C++

# Max pooling
pool_size = (2, 2)
stride_pool = (2, 2)
# pooled = rv.max_pool2d(output, kernel_size=pool_size, stride=stride_pool)
pooled = rv.max_pooling_2d(output, pool_size[0], pool_size[1], stride_pool[0], stride_pool[1]) # Exemplo adaptado

# Batch normalization (exemplo de parâmetros)
channels = pooled.shape[1] # Assumindo NCHW
gamma = np.random.uniform(0.5, 1.5, channels).astype(np.float32)
beta = np.random.uniform(-0.5, 0.5, channels).astype(np.float32)
# mean e var podem ser necessários se a função não os calcular internamente
mean = np.zeros(channels, dtype=np.float32)
var = np.ones(channels, dtype=np.float32)
epsilon = 1e-5

# normalized = rv.batch_norm(pooled, gamma, beta, mean, var, epsilon) # Nome hipotético
# Verifique a assinatura exata da função na API Python

# Softmax (exemplo em um vetor achatado)
logits = np.random.uniform(-5, 5, 10).astype(np.float32)
# probabilities = rv.softmax(logits) # Nome hipotético
```

*Nota: Os nomes exatos das funções Python e seus parâmetros podem variar. Consulte a documentação da API Python da biblioteca.*

## Benchmarks Comparativos

A biblioteca RVV-SIMD inclui um conjunto abrangente de benchmarks para avaliar o desempenho das operações otimizadas para RVV e compará-las com:

1.  Implementações escalares de fallback (baseline).
2.  Implementações equivalentes usando extensões SIMD de outras arquiteturas (requer compilação e execução em hardware correspondente):
    *   **x86**: AVX, AVX2, AVX-512
    *   **ARM**: NEON

### Metodologia de Benchmark

Os benchmarks medem o desempenho de operações chave em diferentes tamanhos de dados para avaliar:

*   **Throughput**: Quantidade de dados processados por segundo (e.g., GFLOPS para operações de ponto flutuante, GB/s para movimentação de dados).
*   **Latência**: Tempo médio para completar uma única operação ou um lote pequeno.
*   **Speedup**: Aceleração relativa da implementação RVV em comparação com a implementação escalar de fallback e, potencialmente, com implementações AVX/NEON.

As categorias de benchmark incluem:

1.  **Operações Vetoriais Core**: Adição, multiplicação, produto escalar, funções matemáticas, etc.
2.  **Operações Matriciais**: Adição, multiplicação elemento-a-elemento, multiplicação de matrizes (GEMM), transposição.
3.  **Operações de Machine Learning**: Convolução 2D, pooling, batch normalization, softmax.

### Executando Benchmarks

Após compilar a biblioteca com sucesso:

```bash
cd build
# Certifique-se que o target de benchmark foi compilado (pode ter um nome específico)
make rvv_simd_benchmarks # Ou o nome real do target
# Execute o binário de benchmark
./benchmarks/rvv_simd_benchmarks --benchmark_filter=all # Executa todos os benchmarks
# ./benchmarks/rvv_simd_benchmarks --benchmark_filter=VectorAdd # Executa benchmarks específicos
```

Os benchmarks geralmente utilizam bibliotecas como Google Benchmark para fornecer resultados detalhados.

### Exemplo de Resultados (Formato Ilustrativo)

```
--------------------------------------------------------------------
Benchmark                              Time           CPU Iterations
--------------------------------------------------------------------
BM_VectorAdd_RVV/1024              100 ns        100 ns   7000000
BM_VectorAdd_Fallback/1024        1000 ns       1000 ns    700000
BM_MatrixMul_RVV/32x64x32         5000 ns       5000 ns     140000
BM_MatrixMul_Fallback/32x64x32   50000 ns      50000 ns      14000

Comparação de Desempenho: Produto Escalar (1M elementos float32)
| Arquitetura       | Tempo (ms) | Speedup vs Fallback | GFLOPS |
|-------------------|------------|---------------------|--------|
| RISC-V (RVV)      |   1.2      |        25.0x        |  1.67  |
| x86 (AVX2)        |   0.8      |        37.5x        |  2.50  | # Exemplo
| ARM (NEON)        |   2.5      |        12.0x        |  0.80  | # Exemplo
| Fallback (Escalar)|  30.0      |         1.0x        |  0.07  |
|-------------------|------------|---------------------|--------|
```

*Nota: Os resultados reais dependerão do hardware específico, compilador, e tamanho dos dados.*

## Aplicações em Machine Learning

A RVV-SIMD é projetada para acelerar componentes computacionalmente intensivos de algoritmos de Machine Learning, especialmente em plataformas RISC-V embarcadas ou servidores onde a eficiência é crucial.

### Redes Neurais Convolucionais (CNNs)

Operações fundamentais em CNNs, como convoluções e pooling, são inerentemente paralelas e se beneficiam enormemente da vetorização RVV:

*   **Convolução 2D**: Acelerada através de algoritmos como `im2col` + GEMM ou abordagens diretas otimizadas com instruções RVV (e.g., `vfmacc`).
*   **Pooling (Max/Average)**: Implementações eficientes usando comparações e reduções vetoriais.
*   **Batch Normalization**: Operações vetoriais de adição, multiplicação e divisão aplicadas aos canais.
*   **Funções de Ativação**: Aplicação elemento-a-elemento de funções como ReLU, Sigmoid, Tanh usando instruções vetoriais.
*   **Softmax**: Combinação de exponenciação vetorial e redução de soma.

### Outros Algoritmos

Além de CNNs, outras áreas de ML podem se beneficiar:

*   **Processamento de Linguagem Natural (NLP)**: Multiplicação de matrizes em Transformers, operações em embeddings.
*   **Redes Neurais Recorrentes (RNNs)**: Multiplicações matriz-vetor.
*   **Algoritmos de Clusterização/Distância**: Cálculo de distâncias (Euclidiana, cosseno) entre vetores.
*   **Processamento de Sinais**: FFT, filtros aplicados a dados de sensores.

### Exemplo: Forward Pass de uma CNN Simples (Python)

Este exemplo ilustra como as funções da RVV-SIMD podem ser usadas para construir as camadas de uma CNN simples.

```python
import numpy as np
import rvv_simd as rv

# --- Definição Hipotética de Camadas usando RVV-SIMD ---
class ConvLayer:
    def __init__(self, kernel, stride=1, padding=0):
        self.kernel = kernel.astype(np.float32)
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        # Assume x é NCHW, kernel é OutCHW
        # A API Python pode precisar de ajustes no layout ou parâmetros
        # Exemplo simplificado para batch=1
        if x.ndim == 4: x = x[0] # Processa uma imagem por vez
        return rv.convolution_2d(x, self.kernel,
                                 self.stride, self.stride,
                                 self.padding, self.padding) # Adapte à API real

class ReLULayer:
    def forward(self, x):
        # Assume que rv.relu opera em todo o tensor ou precisa de loop
        if x.ndim == 3: # CHW
           result = np.zeros_like(x)
           for c in range(x.shape[0]):
               result[c] = rv.relu(x[c].flatten()).reshape(x.shape[1], x.shape[2])
           return result
        elif x.ndim == 1: # Vetor
            return rv.relu(x)
        else:
             raise ValueError("ReLU input shape not supported")


class MaxPoolLayer:
    def __init__(self, kernel_size=2, stride=2):
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
         return rv.max_pooling_2d(x, self.kernel_size, self.kernel_size,
                                  self.stride, self.stride) # Adapte à API real

class FlattenLayer:
    def forward(self, x):
        self.input_shape = x.shape
        return x.flatten()

class LinearLayer:
    def __init__(self, weights, bias):
        self.weights = weights.astype(np.float32) # Out, In
        self.bias = bias.astype(np.float32)       # Out

    def forward(self, x):
        # x é um vetor (In,)
        # matmul(W, x.T) -> (Out, 1) -> flatten -> (Out,)
        output = rv.matrix_mul(self.weights, x.reshape(-1, 1)).flatten()
        output = rv.vector_add(output, self.bias)
        return output

class SoftmaxLayer:
     def forward(self, x):
         # Assume rv.softmax opera em vetor
         return rv.softmax(x)


# --- Construção e Execução da Rede ---
# Parâmetros (exemplo)
input_c, input_h, input_w = 3, 32, 32
conv1_f, conv1_k = 16, 3
pool_k = 2
conv2_f, conv2_k = 32, 3
fc1_size = 128
num_classes = 10

# Pesos e Biases (inicialização aleatória para exemplo)
k1 = np.random.randn(conv1_f, input_c, conv1_k, conv1_k).astype(np.float32)
k2 = np.random.randn(conv2_f, conv1_f, conv2_k, conv2_k).astype(np.float32)
# Calcular tamanho após pooling2
final_h = (input_h // pool_k // pool_k)
final_w = (input_w // pool_k // pool_k)
fc1_in_size = conv2_f * final_h * final_w
w1 = np.random.randn(fc1_size, fc1_in_size).astype(np.float32)
b1 = np.random.randn(fc1_size).astype(np.float32)
w_out = np.random.randn(num_classes, fc1_size).astype(np.float32)
b_out = np.random.randn(num_classes).astype(np.float32)

# Criação das camadas
layer1_conv = ConvLayer(k1, padding=1)
layer1_relu = ReLULayer()
layer1_pool = MaxPoolLayer(kernel_size=pool_k, stride=pool_k)
layer2_conv = ConvLayer(k2, padding=1)
layer2_relu = ReLULayer()
layer2_pool = MaxPoolLayer(kernel_size=pool_k, stride=pool_k)
flatten = FlattenLayer()
layer3_fc = LinearLayer(w1, b1)
layer3_relu = ReLULayer()
layer_out = LinearLayer(w_out, b_out)
softmax = SoftmaxLayer()

# Forward Pass
input_data = np.random.randn(1, input_c, input_h, input_w).astype(np.float32) # Batch = 1

x = layer1_conv.forward(input_data)
x = layer1_relu.forward(x)
x = layer1_pool.forward(x)
x = layer2_conv.forward(x)
x = layer2_relu.forward(x)
x = layer2_pool.forward(x)
x = flatten.forward(x)
x = layer3_fc.forward(x)
x = layer3_relu.forward(x)
logits = layer_out.forward(x)
probabilities = softmax.forward(logits)

print("Formato da Saída (Convolução 1):", layer1_conv.forward(input_data).shape)
print("Formato da Saída (Pooling 1):", layer1_pool.forward(layer1_relu.forward(layer1_conv.forward(input_data))).shape)
print("Formato da Saída (Pooling 2):", x.shape) # Após Pool2
print("Logits:", logits)
print("Probabilidades:", probabilities)
```

## Otimizações para RVV

A biblioteca RVV-SIMD emprega várias técnicas de otimização específicas da extensão vetorial RISC-V para alcançar alto desempenho:

### 1. Adaptação Dinâmica do Comprimento Vetorial (VLA - Vector Length Agnostic)

A RVV permite que o software ajuste o número de elementos processados por instrução vetorial (`vl`) em tempo de execução, até o máximo suportado pelo hardware (`VLEN`). A biblioteca utiliza `vsetvl` (ou `__riscv_vsetvl_e<ew>m<lmul>` intrínseco) no início de loops para processar dados em blocos de tamanho ideal, garantindo portabilidade e eficiência em diferentes implementações de RVV.

```cpp
// Exemplo em um loop de adição vetorial
size_t remaining_length = length;
float *pa = a, *pb = b, *pc = result;

while (remaining_length > 0) {
    size_t vl = __riscv_vsetvl_e32m8(remaining_length); // Define vl para o loop atual (até M8*VLEN/32 elementos)
    vfloat32m8_t va = __riscv_vle32_v_f32m8(pa, vl);    // Carrega vl elementos de a
    vfloat32m8_t vb = __riscv_vle32_v_f32m8(pb, vl);    // Carrega vl elementos de b
    vfloat32m8_t vc = __riscv_vfadd_vv_f32m8(va, vb, vl); // Adiciona vl elementos
    __riscv_vse32_v_f32m8(pc, vc, vl);                 // Armazena vl elementos em c

    pa += vl; // Avança ponteiros
    pb += vl;
    pc += vl;
    remaining_length -= vl; // Decrementa contador
}
```

### 2. Padrões Eficientes de Acesso à Memória

As instruções RVV suportam diferentes modos de acesso à memória:

*   **Unit-stride**: Carrega/armazena elementos contíguos (usado na maioria das operações vetoriais básicas).
*   **Strided**: Acessa elementos com um passo constante (útil em certas operações matriciais ou de processamento de sinais).
*   **Indexed**: Acessa elementos usando um vetor de índices (útil para gather/scatter).

A biblioteca prioriza acessos *unit-stride* sempre que possível, pois geralmente são os mais rápidos. Algoritmos como a multiplicação de matrizes ou convoluções podem ser reestruturados (e.g., `im2col`) para maximizar o acesso sequencial.

### 3. Operações de Redução Otimizadas

Operações que agregam valores de um vetor (soma, máximo, mínimo, produto escalar) utilizam instruções de redução vetorial dedicadas (`vfredusum`, `vfredmax`, etc.). Essas instruções são significativamente mais eficientes do que um loop escalar sobre os elementos do vetor.

```cpp
// Exemplo de redução de soma para produto escalar (simplificado)
vfloat32m1_t vsum_res = __riscv_vfmv_s_f_f32m1(0.0f); // Inicializa acumulador (LMUL=1)

size_t remaining_length = length;
float *pa = a, *pb = b;

while (remaining_length > 0) {
    size_t vl = __riscv_vsetvl_e32m8(remaining_length); // Usa LMUL=8 para operações internas
    vfloat32m8_t va = __riscv_vle32_v_f32m8(pa, vl);
    vfloat32m8_t vb = __riscv_vle32_v_f32m8(pb, vl);
    // Acumulação vetorial usando vfmacc (Multiply-Accumulate) ou vfmul + vfredusum
    // Exemplo com vfmul + vfredusum:
    vfloat32m8_t vprod = __riscv_vfmul_vv_f32m8(va, vb, vl);
    vsum_res = __riscv_vfredusum_vs_f32m8_f32m1(vprod, vsum_res, vl); // Acumula no registrador LMUL=1

    pa += vl;
    pb += vl;
    remaining_length -= vl;
}
float final_sum = __riscv_vfmv_f_s_f32m1_f32(vsum_res); // Extrai resultado final
```

### 4. Agrupamento de Registradores Vetoriais (LMUL)

O LMUL (Length Multiplier) permite tratar múltiplos registradores vetoriais como um único registrador lógico maior (LMUL=2, 4, 8) ou usar frações de um registrador (LMUL=1/2, 1/4, 1/8). A biblioteca pode usar LMUL > 1 para:

*   Aumentar o número de elementos processados por iteração em loops.
*   Reduzir a sobrecarga de controle de loop.
*   Manter mais dados intermediários em registradores (útil em reduções ou algoritmos complexos).

A escolha do LMUL ideal depende da operação específica, da disponibilidade de registradores e do `VLEN` do hardware.

### 5. Mascaramento (Predicação)

Instruções vetoriais podem operar condicionalmente em elementos individuais usando um registrador de máscara (geralmente `v0`). Isso é útil para:

*   Implementar lógica condicional (`if`/`else`) em nível de elemento sem ramificações custosas.
*   Lidar com elementos de borda em loops sem código separado.

```cpp
// Exemplo hipotético: result[i] = (a[i] > 0) ? b[i] : c[i];
// vbool4_t mask = __riscv_vmflt_vf_f32m8_b4(va, 0.0f, vl); // Gera máscara onde a[i] > 0
// vfloat32m8_t vresult = __riscv_vmerge_vvm_f32m8(vc, vb, mask, vl); // Mescla b[i] (onde mask=1) e c[i] (onde mask=0)
```

Estas otimizações, combinadas, permitem que a RVV-SIMD extraia um desempenho significativo do hardware RISC-V com extensão vetorial.

## Alternativas e Comparação

Existem outras abordagens para utilizar SIMD em RISC-V:

### 1. SIMD Everywhere (SIMDe)

*   **Abordagem**: Biblioteca de cabeçalhos C que emula APIs SIMD de outras arquiteturas (SSE, AVX, NEON) usando C puro ou, quando disponível, os intrínsecos da arquitetura alvo (incluindo RVV).
*   **Vantagens**:
    *   Alta portabilidade: Permite compilar código SIMD existente (escrito para x86/ARM) em RISC-V.
    *   Reduz o esforço de migração de código legado.
*   **Desvantagens**:
    *   A tradução pode não gerar o código RVV mais otimizado.
    *   Pode não explorar totalmente os recursos exclusivos da RVV (como VLA de forma ideal).
    *   A performance depende da qualidade da emulação/tradução para RVV.
*   **Referência**: [Arxiv: Bringing SIMD Everywhere via Automatic Translation](https://arxiv.org/abs/2309.16509)

### 2. `rvv::experimental::simd` (Exemplo de Biblioteca Nativa)

*   **Abordagem**: Biblioteca C++ que fornece uma API de alto nível, baseada em templates e inspirada no padrão C++ Parallelism TS (`std::experimental::simd`), especificamente para RVV.
*   **Vantagens**:
    *   API moderna e expressiva em C++.
    *   Abstração sobre os intrínsecos RVV.
    *   Potencialmente gera código RVV otimizado através de templates.
*   **Desvantagens**:
    *   Específica para RVV (não portável para x86/ARM sem reescrita).
    *   Pode ser experimental ou ter menos funcionalidades que bibliotecas maduras.
*   **Referência**: [GitHub: Pansysk75/cpp-simd-riscv](https://github.com/Pansysk75/cpp-simd-riscv) (Exemplo de implementação)

### 3. Extensão RISC-V "P" (Packed SIMD)

*   **Abordagem**: Uma extensão RISC-V *diferente* da "V" (Vector). Define operações SIMD de largura fixa (e.g., 32, 64 bits) sobre os registradores inteiros ou de ponto flutuante existentes. Mais similar ao MMX/SSE inicial ou NEON básico.
*   **Vantagens**:
    *   Potencialmente mais simples de implementar em hardware de baixo custo.
    *   API pode ser mais simples que RVV.
*   **Desvantagens**:
    *   Largura fixa (não VLA), menos flexível e escalável que RVV.
    *   Menor poder computacional por instrução comparado a RVV com VLENs maiores.
    *   Ecossistema e suporte de ferramentas ainda em desenvolvimento.
*   **Referência**: [GitHub: riscv/riscv-p-spec](https://github.com/riscv/riscv-p-spec)

### Comparação RVV-SIMD vs. Alternativas

| Critério             | RVV-SIMD (Esta Biblioteca)              | SIMDe                                   | `rvv::simd` (Exemplo)                   | Extensão "P"          |
| :------------------- | :-------------------------------------- | :-------------------------------------- | :-------------------------------------- | :-------------------- |
| **Foco Principal**   | Otimização RVV Nativa + Python          | Portabilidade de Código SIMD Existente | API C++ Moderna para RVV                | SIMD de Largura Fixa  |
| **Portabilidade**    | RISC-V (RVV) + Fallback Escalar         | Multi-arquitetura (x86, ARM, RVV)       | Específico RVV                          | Específico RISC-V "P" |
| **Nível Abstração**  | Médio (API C/Python sobre intrínsecos) | Baixo (APIs de Intrínsecos Emuladas)    | Alto (Templates C++)                    | Baixo (Intrínsecos)   |
| **Desempenho RVV**   | **Alto (Otimizado)**                    | Variável (Depende da Tradução)          | Potencialmente Alto                     | Moderado (Largura Fixa) |
| **Interface Python** | **Sim (pybind11)**                      | Não Diretamente (Via libs C/C++)        | Não Diretamente                         | Não Diretamente       |
| **Madurez**          | (Definido pelo Projeto)                 | Produção                                | Experimental                            | Em Desenvolvimento    |

**RVV-SIMD** se posiciona como uma solução focada em extrair o máximo desempenho da extensão RVV, oferecendo interfaces C++ e Python convenientes, sacrificando a portabilidade direta de código SIMD de outras arquiteturas que o SIMDe oferece.

## Estado Atual do Suporte Python (Considerações)

A integração direta e eficiente de extensões SIMD de baixo nível como a RVV com uma linguagem de alto nível como Python apresenta desafios e o suporte atual é limitado:

1.  **Simuladores Python**: Simuladores RISC-V escritos puramente em Python, como `riscvsim.py`, geralmente **não suportam** a extensão vetorial devido à complexidade de modelar `vl`, `vtype`, LMUL, mascaramento e o grande conjunto de instruções vetoriais de forma precisa e performática em Python.
2.  **Complexidade Arquitetural**: A natureza VLA (Vector Length Agnostic) e configurável da RVV (LMUL, SEW) torna a simulação ou a interface direta mais complexa do que SIMD de largura fixa.
3.  **Desempenho**: Chamar operações vetoriais individuais a partir de Python puro teria uma sobrecarga significativa. O desempenho real vem da execução de sequências otimizadas de instruções RVV em código nativo (C/C++), como feito nesta biblioteca RVV-SIMD.
4.  **Abordagem Prática (Bindings)**: A abordagem mais viável e performática, adotada por esta biblioteca, é implementar o núcleo otimizado em C/C++ usando intrínsecos RVV e expor essa funcionalidade para Python através de bindings (e.g., `pybind11`, `Cython`). Isso permite que código Python de alto nível utilize operações vetoriais aceleradas que rodam nativamente.
5.  **Ecossistema**: O suporte RVV em compiladores (GCC, LLVM) e bibliotecas de baixo nível está amadurecendo. Ferramentas Python que dependem dessas cadeias de ferramentas se beneficiarão indiretamente. Bibliotecas como NumPy/SciPy não geram código RVV diretamente, mas podem ser compiladas com um toolchain RISC-V que pode otimizar *algumas* operações internamente se o compilador suportar auto-vetorização para RVV (o que ainda é uma área em desenvolvimento).

Portanto, embora *simular* RVV em Python puro seja limitado, *utilizar* operações RVV otimizadas a partir de Python é viável e é o objetivo principal dos bindings Python desta biblioteca.

## Estrutura do Projeto

```
rvv-simd/
├── src/                    # Código fonte C/C++ da biblioteca core
│   ├── core/               # Operações SIMD vetoriais e matriciais básicas
│   ├── ml/                 # Operações específicas de Machine Learning (Conv, Pool, etc.)
│   └── common/             # Funções utilitárias, tipos, detecção de RVV
├── include/                # Arquivos de cabeçalho públicos (.h ou .hpp)
│   └── rvv_simd/           # Cabeçalhos organizados por módulo
├── python/                 # Bindings e pacote Python
│   ├── src/                # Código C++ para os bindings (usando pybind11)
│   ├── rvv_simd/           # Código Python do pacote (se houver, __init__.py etc.)
│   ├── examples/           # Exemplos de uso em Python
│   └── setup.py            # Script para construir e instalar o pacote Python
├── benchmarks/             # Código para os benchmarks de desempenho
│   ├── core/               # Benchmarks de operações core
│   ├── ml/                 # Benchmarks de operações ML
│   └── common/             # Utilitários para benchmarking
├── tests/                  # Testes unitários e de integração (e.g., usando Google Test)
│   ├── core/
│   ├── ml/
│   └── python/             # Testes para os bindings Python (e.g., usando pytest)
├── docs/                   # Documentação (gerada por Doxygen, Sphinx, etc.)
├── examples/               # Exemplos de uso da biblioteca em C++
├── cmake/                  # Módulos CMake customizados (se houver)
├── LICENSE                 # Arquivo de licença (e.g., MIT)
├── README.md               # Este arquivo
├── CONTRIBUTING.md         # Diretrizes para contribuição
└── CMakeLists.txt          # Script principal de build CMake
```

## Contribuições

Contribuições para a biblioteca RVV-SIMD são muito bem-vindas! Se você deseja contribuir, por favor:

1.  Verifique as [Issues](https://github.com/<your_username>/rvv-simd/issues) abertas para tarefas existentes ou relate novos bugs/sugestões.
2.  Faça um Fork do repositório.
3.  Crie um branch para sua feature ou correção (`git checkout -b feature/nova-operacao` ou `fix/bug-xyz`).
4.  Implemente suas mudanças e adicione testes apropriados.
5.  Certifique-se que os testes passam (`make test` ou `pytest`).
6.  Faça o commit das suas mudanças (`git commit -m 'Adiciona nova operação X'`).
7.  Faça o Push para o seu fork (`git push origin feature/nova-operacao`).
8.  Abra um Pull Request para o repositório principal.

Por favor, siga as diretrizes detalhadas em [CONTRIBUTING.md](CONTRIBUTING.md) (crie este arquivo se necessário).

## Licença

Este projeto é licenciado sob os termos da **Licença MIT**. Veja o arquivo [LICENSE](LICENSE) para detalhes completos.

## Agradecimentos

*   À comunidade RISC-V International pelo desenvolvimento da arquitetura e da extensão vetorial.
*   Aos desenvolvedores das toolchains GCC e LLVM pelo suporte à compilação para RISC-V e RVV.
*   Aos criadores de `pybind11` por facilitar a criação de bindings Python.
*   A todos os contribuidores e usuários desta biblioteca.
```