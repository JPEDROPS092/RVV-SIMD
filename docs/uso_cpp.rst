Uso da Biblioteca em C++
=====================

Esta seção descreve como usar a biblioteca RVV-SIMD em aplicações C++.

Incluindo a Biblioteca
---------------------

Para usar a biblioteca RVV-SIMD em seu projeto C++, você precisa incluir os cabeçalhos apropriados:

.. code-block:: cpp

    #include "rvv_simd.h" // Cabeçalho principal
    
    // Ou incluir cabeçalhos específicos
    #include "rvv_simd/core.h"      // Funções core
    #include "rvv_simd/vector_ops.h" // Operações vetoriais
    #include "rvv_simd/matrix_ops.h" // Operações matriciais
    #include "rvv_simd/ml_ops.h"     // Operações de ML

Inicializando e Verificando Suporte
----------------------------------

Antes de usar a biblioteca, é recomendável inicializá-la e verificar se a extensão vetorial RISC-V (RVV) é suportada:

.. code-block:: cpp

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

Operações Vetoriais
------------------

A biblioteca fornece várias operações vetoriais otimizadas:

.. code-block:: cpp

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

Operações Matriciais
------------------

A biblioteca também fornece operações matriciais otimizadas:

.. code-block:: cpp

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

Operações de Machine Learning
---------------------------

Para aplicações de machine learning, a biblioteca fornece operações como convolução, pooling, etc.:

.. code-block:: cpp

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

Exemplo Completo
--------------

Aqui está um exemplo completo de como usar a biblioteca para realizar uma operação de multiplicação de matrizes:

.. code-block:: cpp

    #include <iostream>
    #include <vector>
    #include <random>
    #include <chrono>
    #include "rvv_simd.h"

    int main() {
        // Inicializa a biblioteca
        if (!rvv_simd::initialize()) {
            std::cerr << "Falha ao inicializar a biblioteca RVV-SIMD" << std::endl;
            return 1;
        }

        // Verifica se RVV é suportado
        std::cout << "RVV suportado: " << (rvv_simd::is_rvv_supported() ? "Sim" : "Não") << std::endl;
        std::cout << "Versão RVV-SIMD: " << rvv_simd::get_version() << std::endl;

        // Define dimensões das matrizes
        const size_t a_rows = 128, a_cols = 256;
        const size_t b_rows = a_cols, b_cols = 64;
        const size_t c_rows = a_rows, c_cols = b_cols;

        // Aloca memória para as matrizes
        std::vector<float> a(a_rows * a_cols);
        std::vector<float> b(b_rows * b_cols);
        std::vector<float> c(c_rows * c_cols);

        // Inicializa matrizes com valores aleatórios
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        for (size_t i = 0; i < a.size(); ++i) {
            a[i] = dist(gen);
        }

        for (size_t i = 0; i < b.size(); ++i) {
            b[i] = dist(gen);
        }

        // Mede o tempo de execução
        auto start = std::chrono::high_resolution_clock::now();

        // Realiza multiplicação de matrizes
        rvv_simd::matrix_mul(a.data(), b.data(), a_rows, a_cols, b_cols, c.data());

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;

        std::cout << "Multiplicação de matrizes " << a_rows << "x" << a_cols << " * " 
                  << b_rows << "x" << b_cols << " concluída em " 
                  << duration.count() << " ms" << std::endl;

        // Exibe alguns elementos do resultado
        std::cout << "Primeiros elementos do resultado:" << std::endl;
        for (size_t i = 0; i < std::min(size_t(5), c_rows); ++i) {
            for (size_t j = 0; j < std::min(size_t(5), c_cols); ++j) {
                std::cout << c[i * c_cols + j] << " ";
            }
            std::cout << std::endl;
        }

        return 0;
    }

Compilando e Executando
---------------------

Para compilar um programa que usa a biblioteca RVV-SIMD, você precisa incluir os cabeçalhos e linkar com a biblioteca:

.. code-block:: bash

    # Compilando com GCC
    g++ -std=c++14 -I/caminho/para/rvv-simd/include -L/caminho/para/rvv-simd/lib -o meu_programa meu_programa.cpp -lrvv_simd
    
    # Compilando com CMake
    # No CMakeLists.txt:
    # find_package(RVV_SIMD REQUIRED)
    # target_link_libraries(meu_programa PRIVATE RVV_SIMD::rvv_simd)

Considerações de Desempenho
-------------------------

Para obter o melhor desempenho da biblioteca RVV-SIMD, considere as seguintes dicas:

1. **Alinhamento de memória**: Alinhe seus dados em limites de cache para melhor desempenho.
2. **Reutilização de buffers**: Reutilize buffers de resultado para evitar alocações de memória desnecessárias.
3. **Tamanho dos dados**: Operações em vetores/matrizes maiores geralmente têm melhor desempenho relativo devido à sobrecarga de inicialização.
4. **Precisão**: A biblioteca é otimizada para operações em ponto flutuante de precisão simples (float32).
5. **Compilação**: Use flags de otimização apropriadas ao compilar seu programa (e.g., `-O3`).