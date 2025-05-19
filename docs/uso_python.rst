Uso da Biblioteca em Python
=======================

Esta seção descreve como usar a biblioteca RVV-SIMD em aplicações Python.

Importando a Biblioteca
----------------------

Para usar a biblioteca RVV-SIMD em seu projeto Python, você precisa importá-la:

.. code-block:: python

    import numpy as np
    import rvv_simd as rv
    
    # Verifica se a versão com RVV está ativa (se aplicável)
    print(f"Versão RVV-SIMD: {rv.get_version()}")
    print(f"Suporte a RVV: {'Sim' if rv.is_rvv_supported() else 'Não'}")
    if rv.is_rvv_supported():
        print(f"Informações RVV: {rv.get_rvv_info()}")

Operações Vetoriais
------------------

A biblioteca fornece várias operações vetoriais otimizadas:

.. code-block:: python

    # Cria vetores NumPy (precisão float32 é comum)
    size = 1024
    a = np.random.uniform(-10, 10, size).astype(np.float32)
    b = np.random.uniform(-10, 10, size).astype(np.float32)
    
    # Adição de vetores (aceita e retorna NumPy arrays)
    # Usando a API de baixo nível
    c1 = rv.vector_add(a, b)
    
    # Usando a API estilo NumPy (recomendado)
    c2 = rv.add(a, b)
    
    # Produto escalar
    dot1 = rv.vector_dot(a, b)
    dot2 = rv.dot(a, b)  # API estilo NumPy
    
    # Escalonamento de vetor
    scaled = rv.scale(a, 2.5)
    
    # Normalização de vetor
    normalized = rv.normalize(a)
    
    # Funções de ativação
    sigmoid_result = rv.sigmoid(a)
    relu_result = rv.relu(a)
    tanh_result = rv.tanh(a)

Operações Matriciais
------------------

A biblioteca também fornece operações matriciais otimizadas:

.. code-block:: python

    # Cria matrizes NumPy
    rows, cols = 32, 32
    a = np.random.uniform(-10, 10, (rows, cols)).astype(np.float32)
    b = np.random.uniform(-10, 10, (rows, cols)).astype(np.float32)
    
    # Adição de matrizes
    c = rv.add(a, b)  # Funciona tanto para vetores quanto para matrizes
    
    # Multiplicação de matrizes
    a_rows, a_cols, b_cols = 32, 64, 32
    a_mat = np.random.uniform(-10, 10, (a_rows, a_cols)).astype(np.float32)
    b_mat = np.random.uniform(-10, 10, (a_cols, b_cols)).astype(np.float32)
    
    # Usando a API de baixo nível
    c_mat1 = rv.matrix_mul(a_mat, b_mat)
    
    # Usando a API estilo NumPy (recomendado)
    c_mat2 = rv.matmul(a_mat, b_mat)
    
    # Transposição de matriz
    a_transpose = rv.transpose(a)

Operações de Machine Learning
---------------------------

Para aplicações de machine learning, a biblioteca fornece operações como convolução, pooling, etc.:

.. code-block:: python

    # Cria tensor de entrada (NCHW - batch, canais, altura, largura)
    batch_size = 1
    input_channels = 3
    input_height = 32
    input_width = 32
    input_tensor = np.random.uniform(-1, 1, (batch_size, input_channels, input_height, input_width)).astype(np.float32)
    
    # Cria tensor de kernel (NCHW - num_kernels, canais_in, altura, largura)
    kernel_num = 16
    kernel_height = 3
    kernel_width = 3
    kernel_tensor = np.random.uniform(-1, 1, (kernel_num, input_channels, kernel_height, kernel_width)).astype(np.float32)
    
    # Operação de convolução
    stride = (1, 1)
    padding = (1, 1)
    
    # Usando a API estilo NumPy (recomendado)
    output = rv.conv2d(input_tensor, kernel_tensor, stride=stride, padding=padding)
    
    # Usando a API de baixo nível
    # output = rv.convolution_2d(input_tensor[0], kernel_tensor, stride[0], stride[1], padding[0], padding[1])
    
    # Max pooling
    pool_size = (2, 2)
    stride_pool = (2, 2)
    
    # Usando a API estilo NumPy (recomendado)
    pooled = rv.max_pool2d(output, kernel_size=pool_size, stride=stride_pool)
    
    # Usando a API de baixo nível
    # pooled = rv.max_pooling_2d(output, pool_size[0], pool_size[1], stride_pool[0], stride_pool[1])

Exemplo Completo
--------------

Aqui está um exemplo completo de como usar a biblioteca para implementar uma camada densa (fully connected) de uma rede neural:

.. code-block:: python

    import numpy as np
    import rvv_simd as rv
    import time
    
    # Implementação simplificada de uma camada densa (fully connected)
    def dense_layer(input_data, weights, bias, activation='relu'):
        # Multiplicação de matriz: output = input * weights + bias
        output = rv.matmul(input_data, weights)
        output = rv.add(output, bias)
        
        # Aplicar função de ativação
        if activation == 'relu':
            return rv.relu(output)
        elif activation == 'sigmoid':
            return rv.sigmoid(output)
        elif activation == 'tanh':
            return rv.tanh(output)
        else:
            return output
    
    # Exemplo de uso
    batch_size = 32
    input_features = 128
    output_features = 64
    
    # Dados de entrada e parâmetros
    input_data = np.random.uniform(-1, 1, (batch_size, input_features)).astype(np.float32)
    weights = np.random.uniform(-0.1, 0.1, (input_features, output_features)).astype(np.float32)
    bias = np.random.uniform(-0.1, 0.1, output_features).astype(np.float32)
    
    # Mede o tempo de execução
    start_time = time.time()
    
    # Executar a camada
    output = dense_layer(input_data, weights, bias, activation='relu')
    
    end_time = time.time()
    
    print(f"Tempo de execução: {(end_time - start_time) * 1000:.2f} ms")
    print(f"Saída: shape={output.shape}, min={output.min()}, max={output.max()}")
    
    # Comparação com NumPy puro
    start_time = time.time()
    
    # Implementação NumPy
    np_output = np.maximum(0, np.dot(input_data, weights) + bias)
    
    end_time = time.time()
    
    print(f"Tempo NumPy: {(end_time - start_time) * 1000:.2f} ms")
    print(f"Resultado equivalente: {np.allclose(output, np_output)}")

Exemplo de Processamento de Imagem
--------------------------------

Aqui está um exemplo de como usar a biblioteca para aplicar um filtro de convolução a uma imagem:

.. code-block:: python

    import numpy as np
    import rvv_simd as rv
    from PIL import Image
    
    # Carregar imagem e converter para array NumPy
    image = Image.open('exemplo.jpg').convert('RGB')
    img_array = np.array(image).astype(np.float32) / 255.0
    
    # Aplicar filtro de convolução (detecção de bordas)
    kernel = np.array([
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1]
    ], dtype=np.float32)
    
    # Expandir kernel para formato NCHW (1, 1, 3, 3)
    kernel = kernel.reshape(1, 1, 3, 3)
    kernel = np.repeat(kernel, 3, axis=0)  # Um kernel para cada canal
    
    # Preparar imagem para convolução (HWC -> NCHW)
    img_nchw = img_array.transpose(2, 0, 1).reshape(1, 3, img_array.shape[0], img_array.shape[1])
    
    # Aplicar convolução
    result = rv.conv2d(img_nchw, kernel, stride=(1, 1), padding=(1, 1))
    
    # Converter resultado de volta para formato de imagem (NCHW -> HWC)
    result = result[0].transpose(1, 2, 0)
    result = np.clip(result, 0, 1)  # Limitar valores entre 0 e 1
    
    # Salvar resultado
    result_img = Image.fromarray((result * 255).astype(np.uint8))
    result_img.save('resultado.jpg')

Benchmark Comparativo
-------------------

Você pode comparar o desempenho da biblioteca RVV-SIMD com o NumPy:

.. code-block:: python

    import numpy as np
    import rvv_simd as rv
    import time
    import matplotlib.pyplot as plt
    
    # Função para medir o tempo de execução
    def benchmark(func, *args, n_runs=10):
        times = []
        for _ in range(n_runs):
            start_time = time.time()
            result = func(*args)
            times.append(time.time() - start_time)
        return result, sum(times) / n_runs
    
    # Tamanhos de vetores para teste
    sizes = [1000, 10000, 100000, 1000000]
    
    # Resultados
    rv_times = []
    np_times = []
    speedups = []
    
    for size in sizes:
        print(f"Testando tamanho {size}...")
        
        # Cria vetores de teste
        a = np.random.uniform(-10, 10, size).astype(np.float32)
        b = np.random.uniform(-10, 10, size).astype(np.float32)
        
        # Benchmark RVV-SIMD
        _, rv_time = benchmark(rv.add, a, b)
        rv_times.append(rv_time)
        
        # Benchmark NumPy
        _, np_time = benchmark(lambda x, y: x + y, a, b)
        np_times.append(np_time)
        
        # Calcula speedup
        speedup = np_time / rv_time
        speedups.append(speedup)
        
        print(f"  RVV-SIMD: {rv_time * 1000:.2f} ms")
        print(f"  NumPy: {np_time * 1000:.2f} ms")
        print(f"  Speedup: {speedup:.2f}x")
    
    # Plota resultados
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(sizes, rv_times, 'o-', label='RVV-SIMD')
    plt.plot(sizes, np_times, 's-', label='NumPy')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Tamanho do Vetor')
    plt.ylabel('Tempo (s)')
    plt.title('Tempo de Execução')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(sizes, speedups, 'D-')
    plt.xscale('log')
    plt.xlabel('Tamanho do Vetor')
    plt.ylabel('Speedup (NumPy / RVV-SIMD)')
    plt.title('Aceleração vs. NumPy')
    plt.axhline(y=1.0, color='r', linestyle='--')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('benchmark.png')
    plt.show()

Considerações de Desempenho
-------------------------

Para obter o melhor desempenho da biblioteca RVV-SIMD em Python, considere as seguintes dicas:

1. **Tipo de dados**: Use sempre `np.float32` para seus arrays NumPy, pois a biblioteca é otimizada para precisão simples.
2. **Contiguidade de memória**: Use arrays contíguos em memória (C-contiguous) para melhor desempenho.
3. **Operações em lote**: Prefira realizar operações em lote em vez de muitas operações pequenas.
4. **Reutilização de arrays**: Reutilize arrays de saída quando possível para evitar alocações de memória desnecessárias.
5. **Evite cópias**: A biblioteca tenta minimizar cópias de dados, mas esteja ciente de que algumas operações podem exigir cópias temporárias.

Integração com Outras Bibliotecas
-------------------------------

A biblioteca RVV-SIMD pode ser integrada com outras bibliotecas Python populares:

.. code-block:: python

    # Integração com PyTorch
    import torch
    import rvv_simd as rv
    
    # Converter tensor PyTorch para NumPy, processar com RVV-SIMD e converter de volta
    def process_tensor(tensor):
        # Converter para NumPy
        np_array = tensor.cpu().numpy()
        
        # Processar com RVV-SIMD
        result = rv.relu(np_array)
        
        # Converter de volta para PyTorch
        return torch.from_numpy(result)
    
    # Integração com TensorFlow
    import tensorflow as tf
    
    # Converter tensor TensorFlow para NumPy, processar com RVV-SIMD e converter de volta
    def process_tf_tensor(tensor):
        # Converter para NumPy
        np_array = tensor.numpy()
        
        # Processar com RVV-SIMD
        result = rv.relu(np_array)
        
        # Converter de volta para TensorFlow
        return tf.convert_to_tensor(result)

Depuração e Solução de Problemas
------------------------------

Se você encontrar problemas ao usar a biblioteca RVV-SIMD em Python, aqui estão algumas dicas de depuração:

1. **Verificar suporte a RVV**:

   .. code-block:: python
   
       import rvv_simd as rv
       print(f"RVV suportado: {'Sim' if rv.is_rvv_supported() else 'Não'}")
       print(f"Versão: {rv.get_version()}")
       print(f"Info: {rv.get_rvv_info()}")

2. **Verificar tipos de dados**:

   .. code-block:: python
   
       import numpy as np
       
       # Certifique-se de que seus arrays são float32
       a = np.array([1, 2, 3], dtype=np.float32)  # Correto
       b = np.array([1, 2, 3])  # Incorreto (será float64 por padrão)
       
       print(f"Tipo de a: {a.dtype}")
       print(f"Tipo de b: {b.dtype}")

3. **Verificar contiguidade de memória**:

   .. code-block:: python
   
       import numpy as np
       
       a = np.random.rand(10, 10).astype(np.float32)
       print(f"C-contiguous: {a.flags.c_contiguous}")
       
       # Se não for contíguo, torne-o contíguo
       if not a.flags.c_contiguous:
           a = np.ascontiguousarray(a)

4. **Comparar resultados com NumPy**:

   .. code-block:: python
   
       import numpy as np
       import rvv_simd as rv
       
       a = np.random.rand(100).astype(np.float32)
       b = np.random.rand(100).astype(np.float32)
       
       # Resultado RVV-SIMD
       c_rv = rv.add(a, b)
       
       # Resultado NumPy
       c_np = a + b
       
       # Comparar
       print(f"Resultados iguais: {np.allclose(c_rv, c_np)}")
       
       # Se não forem iguais, verificar diferenças
       if not np.allclose(c_rv, c_np):
           diff = np.abs(c_rv - c_np)
           print(f"Diferença máxima: {np.max(diff)}")
           print(f"Índice da diferença máxima: {np.argmax(diff)}")
           print(f"Valor RVV-SIMD: {c_rv[np.argmax(diff)]}")
           print(f"Valor NumPy: {c_np[np.argmax(diff)]}")