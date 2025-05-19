Arquitetura
===========

A biblioteca RVV-SIMD é estruturada em camadas para fornecer tanto operações de baixo nível otimizadas quanto interfaces de alto nível para aplicações de ML e processamento de dados.

Visão Geral
-----------

A arquitetura da biblioteca foi projetada com os seguintes objetivos:

* **Desempenho**: Aproveitar ao máximo as capacidades da extensão vetorial RISC-V.
* **Usabilidade**: Fornecer interfaces intuitivas tanto em C++ quanto em Python.
* **Portabilidade**: Funcionar em diferentes implementações de RISC-V e em outras arquiteturas através de implementações de fallback.
* **Extensibilidade**: Facilitar a adição de novas operações e funcionalidades.

Componentes Principais
---------------------

Biblioteca Core (C++)
^^^^^^^^^^^^^^^^^^^^

A biblioteca core é implementada em C++ e consiste nos seguintes componentes:

1. **Operações Vetoriais**:
   * Operações aritméticas básicas (adição, subtração, multiplicação, divisão)
   * Produto escalar (dot product)
   * Escalonamento de vetores
   * Normalização de vetores
   * Funções matemáticas (exp, log, sigmoid, tanh, ReLU)

2. **Operações Matriciais**:
   * Operações aritméticas em matrizes (adição, subtração, multiplicação elemento a elemento)
   * Multiplicação de matrizes
   * Transposição de matrizes
   * Escalonamento de matrizes
   * Normas de matrizes

3. **Operações de Machine Learning**:
   * Operações de convolução para CNNs
   * Operações de pooling (max, average)
   * Batch normalization
   * Funções de ativação (softmax)
   * Funções de perda (cross-entropy)
   * Cálculo de gradientes

Bindings Python
^^^^^^^^^^^^^

Os bindings Python, implementados com `pybind11`, fornecem uma interface de alto nível para a biblioteca core, tornando-a acessível para usuários Python e integrando-a com o ecossistema de ciência de dados do Python:

* Interface compatível com NumPy (aceita e retorna arrays NumPy)
* Suporte para arrays multidimensionais
* Integração facilitada com frameworks de ML do Python (PyTorch, TensorFlow, etc.)
* API intuitiva com nomes de funções familiares para usuários de NumPy

Diagrama de Arquitetura
----------------------

.. code-block:: text

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

Detalhes de Implementação
------------------------

Extensão Vetorial RISC-V (RVV)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A biblioteca utiliza intrínsecos vetoriais RISC-V (`<riscv_vector.h>`) quando compilada para hardware compatível. As otimizações aproveitam a flexibilidade da RVV, como o ajuste dinâmico do comprimento do vetor (`vl`) e o uso eficiente dos registradores vetoriais. Principais intrínsecos utilizados incluem:

* `__riscv_vsetvl_e32m8` (e variantes): Define o comprimento do vetor para processamento.
* `__riscv_vle32_v_f32m8`: Carrega elementos de memória para registradores vetoriais.
* `__riscv_vse32_v_f32m8`: Armazena elementos de registradores vetoriais na memória.
* `__riscv_vfadd_vv_f32m8`, `vfsub`, `vfmul`, `vfdiv`: Operações aritméticas vetoriais.
* `__riscv_vfmacc_vv_f32m8`: Multiplicação-acumulação vetorial (útil em matmul, conv).
* `__riscv_vfredusum_vs_f32m8_f32m1`: Redução de soma vetorial (útil em dot product).
* Operações de máscara para execução condicional.

Implementações de Fallback
^^^^^^^^^^^^^^^^^^^^^^^^

Para garantir a portabilidade e usabilidade em plataformas RISC-V sem a extensão vetorial ou em outras arquiteturas (para fins de teste/comparação), a biblioteca fornece implementações escalares puras em C++ para todas as operações. A seleção entre a implementação RVV e a de fallback é feita em tempo de compilação usando diretivas de pré-processador (`#ifdef __riscv_vector`).

Exemplo de código:

.. code-block:: cpp

    void vector_add(const float* a, const float* b, size_t n, float* result) {
    #ifdef __riscv_vector
        // Implementação usando RVV
        size_t vl;
        for (size_t i = 0; i < n; i += vl) {
            vl = __riscv_vsetvl_e32m8(n - i);
            vfloat32m8_t va = __riscv_vle32_v_f32m8(a + i, vl);
            vfloat32m8_t vb = __riscv_vle32_v_f32m8(b + i, vl);
            vfloat32m8_t vc = __riscv_vfadd_vv_f32m8(va, vb, vl);
            __riscv_vse32_v_f32m8(result + i, vc, vl);
        }
    #else
        // Implementação de fallback
        for (size_t i = 0; i < n; i++) {
            result[i] = a[i] + b[i];
        }
    #endif
    }

Interface Python
--------------

A interface Python é projetada para ser intuitiva e familiar para usuários de NumPy. Ela oferece duas APIs:

1. **API de Baixo Nível**: Funções com prefixos como `vector_*`, `matrix_*` que correspondem diretamente às funções C++
2. **API Estilo NumPy**: Funções com nomes familiares como `add`, `dot`, `matmul` que seguem convenções do NumPy

Exemplo de uso:

.. code-block:: python

    import numpy as np
    import rvv_simd as rv
    
    # Cria vetores NumPy
    a = np.random.uniform(-10, 10, 1000).astype(np.float32)
    b = np.random.uniform(-10, 10, 1000).astype(np.float32)
    
    # API de baixo nível
    c1 = rv.vector_add(a, b)
    
    # API estilo NumPy
    c2 = rv.add(a, b)
    
    # Ambas as chamadas produzem o mesmo resultado
    assert np.allclose(c1, c2)

Gerenciamento de Memória
----------------------

A biblioteca foi projetada para minimizar cópias de memória desnecessárias:

* Em C++, as funções aceitam ponteiros para dados existentes e escrevem resultados em buffers fornecidos pelo usuário.
* Em Python, os bindings utilizam a API de buffer do NumPy para acessar diretamente os dados dos arrays NumPy sem cópias adicionais.

Considerações de Desempenho
-------------------------

Vários fatores foram considerados para otimizar o desempenho:

* **Alinhamento de memória**: Operações vetoriais são mais eficientes quando os dados estão alinhados corretamente.
* **Localidade de cache**: Operações são organizadas para maximizar a localidade de cache.
* **Paralelismo de instrução**: Operações são estruturadas para aproveitar o paralelismo de instrução disponível.
* **Redução de ramificações**: Código vetorial minimiza ramificações condicionais para melhor desempenho.

Extensibilidade
-------------

A biblioteca foi projetada para ser facilmente extensível:

* Novas operações podem ser adicionadas implementando tanto a versão RVV quanto a versão de fallback.
* Os bindings Python podem ser estendidos para expor novas funcionalidades.
* A arquitetura modular facilita a adição de suporte para novas arquiteturas ou otimizações específicas.