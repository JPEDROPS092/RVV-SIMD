Perguntas Frequentes (FAQ)
========================

Esta seção responde às perguntas mais frequentes sobre a biblioteca RVV-SIMD.

Perguntas Gerais
---------------

O que é RVV-SIMD?
^^^^^^^^^^^^^^^

RVV-SIMD é uma biblioteca de computação vetorial otimizada para a extensão vetorial RISC-V (RVV). Ela fornece operações SIMD (Single Instruction, Multiple Data) eficientes para processamento de vetores e matrizes, com foco em aplicações de machine learning e processamento de dados.

Por que usar RVV-SIMD em vez de outras bibliotecas?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A biblioteca RVV-SIMD oferece várias vantagens:

1. **Otimização para RISC-V**: Aproveita ao máximo a extensão vetorial RISC-V, oferecendo desempenho superior em hardware RISC-V.
2. **Portabilidade**: Funciona em diferentes implementações de RISC-V e oferece implementações de fallback para outras arquiteturas.
3. **Bindings Python**: Integração fácil com o ecossistema Python para ciência de dados e machine learning.
4. **Foco em ML**: Operações otimizadas para aplicações de machine learning.

A biblioteca funciona em hardware não-RISC-V?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Sim, a biblioteca inclui implementações de fallback que funcionam em qualquer hardware, incluindo x86 e ARM. No entanto, o desempenho máximo é obtido em hardware RISC-V com suporte à extensão vetorial.

Quais são os requisitos mínimos para usar a biblioteca?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Para C++:
- Compilador C++14 ou superior
- CMake 3.10 ou superior

Para Python:
- Python 3.6 ou superior
- NumPy 1.16 ou superior

Para desempenho máximo:
- Hardware RISC-V com suporte à extensão vetorial (RVV)
- Compilador com suporte a RVV (GCC ou LLVM/Clang)

A biblioteca é de código aberto?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Sim, a biblioteca RVV-SIMD é de código aberto e está disponível sob a licença MIT. Você pode contribuir para o desenvolvimento da biblioteca através do GitHub.

Perguntas Técnicas
-----------------

Quais operações são suportadas pela biblioteca?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A biblioteca suporta uma ampla gama de operações, incluindo:

- Operações vetoriais básicas (adição, subtração, multiplicação, divisão)
- Produto escalar e outras operações de redução
- Operações matriciais (multiplicação, transposição)
- Operações de ML (convolução, pooling, batch normalization)
- Funções de ativação (ReLU, sigmoid, tanh)

Consulte a documentação da API para uma lista completa de operações suportadas.

Como posso verificar se meu hardware suporta RVV?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Em C++:

.. code-block:: cpp

    #include "rvv_simd/core.h"
    
    if (rvv_simd::is_rvv_supported()) {
        std::cout << "RVV é suportado" << std::endl;
        std::cout << "Informações RVV: " << rvv_simd::get_rvv_info() << std::endl;
    } else {
        std::cout << "RVV não é suportado" << std::endl;
    }

Em Python:

.. code-block:: python

    import rvv_simd as rv
    
    if rv.is_rvv_supported():
        print("RVV é suportado")
        print(f"Informações RVV: {rv.get_rvv_info()}")
    else:
        print("RVV não é suportado")

Como posso forçar o uso da implementação de fallback?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Durante a compilação, você pode forçar o uso da implementação de fallback usando a opção `RVV_SIMD_FORCE_FALLBACK`:

.. code-block:: bash

    cmake .. -DRVV_SIMD_FORCE_FALLBACK=ON

Isso é útil para testes ou para comparar o desempenho entre a implementação RVV e a implementação de fallback.

A biblioteca é thread-safe?
^^^^^^^^^^^^^^^^^^^^^^^^^

Sim, a biblioteca é thread-safe para a maioria das operações. As funções que operam em buffers separados podem ser chamadas de diferentes threads sem problemas. No entanto, algumas funções de inicialização e configuração podem não ser thread-safe e devem ser chamadas antes de iniciar threads paralelas.

Quais tipos de dados são suportados?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Atualmente, a biblioteca suporta principalmente operações em ponto flutuante de precisão simples (float32). Algumas operações também suportam inteiros de 32 bits (int32) e ponto flutuante de precisão dupla (float64), mas o suporte é mais limitado.

Em Python, é recomendável usar arrays NumPy com dtype=np.float32 para obter o melhor desempenho.

Como posso contribuir com a biblioteca?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Você pode contribuir de várias maneiras:

1. Reportando bugs e problemas no GitHub
2. Sugerindo novas funcionalidades
3. Enviando pull requests com correções ou novas funcionalidades
4. Melhorando a documentação
5. Escrevendo testes e benchmarks

Consulte o arquivo CONTRIBUTING.md no repositório para mais detalhes.

Perguntas sobre Desempenho
-------------------------

Qual é o ganho de desempenho esperado com RVV?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

O ganho de desempenho depende de vários fatores, incluindo:

- A operação específica sendo executada
- O tamanho dos dados
- A implementação específica de hardware RISC-V
- A largura dos registradores vetoriais disponíveis

Em geral, você pode esperar ganhos de 2x a 10x em comparação com implementações escalares, dependendo da operação e do hardware. Operações como multiplicação de matrizes e convolução tendem a mostrar os maiores ganhos.

Por que algumas operações são mais rápidas que outras?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Algumas operações se beneficiam mais da vetorização do que outras. Operações com alta intensidade aritmética e padrões de acesso à memória regulares (como multiplicação de matrizes) geralmente mostram os maiores ganhos. Operações com muitos acessos aleatórios à memória ou dependências de dados podem se beneficiar menos.

Como posso otimizar meu código para obter o melhor desempenho?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Para obter o melhor desempenho:

1. Use arrays contíguos em memória (C-contiguous)
2. Alinhe seus dados em limites de cache quando possível
3. Use o tipo de dados correto (float32 é geralmente o mais otimizado)
4. Prefira operações em lote em vez de muitas operações pequenas
5. Reutilize buffers de resultado para evitar alocações de memória desnecessárias
6. Considere a localidade de cache ao organizar suas operações

Por que a biblioteca é mais lenta em alguns casos do que NumPy/PyTorch?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Em hardware não-RISC-V, a biblioteca usa implementações de fallback que podem não ser tão otimizadas quanto as implementações específicas de NumPy/PyTorch para x86 (AVX) ou ARM (NEON). Além disso, para operações muito pequenas, a sobrecarga de chamada de função pode dominar o tempo de execução.

Em hardware RISC-V com RVV, a biblioteca deve ser mais rápida que NumPy/PyTorch para a maioria das operações, especialmente para operações de maior intensidade computacional.

Perguntas sobre Integração
------------------------

Como integrar a biblioteca com PyTorch/TensorFlow?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Você pode integrar a biblioteca com PyTorch ou TensorFlow convertendo tensores para arrays NumPy, processando-os com RVV-SIMD e convertendo-os de volta:

.. code-block:: python

    # PyTorch
    import torch
    import rvv_simd as rv
    
    def process_tensor(tensor):
        # Converter para NumPy
        np_array = tensor.cpu().numpy()
        
        # Processar com RVV-SIMD
        result = rv.relu(np_array)
        
        # Converter de volta para PyTorch
        return torch.from_numpy(result)
    
    # TensorFlow
    import tensorflow as tf
    
    def process_tf_tensor(tensor):
        # Converter para NumPy
        np_array = tensor.numpy()
        
        # Processar com RVV-SIMD
        result = rv.relu(np_array)
        
        # Converter de volta para TensorFlow
        return tf.convert_to_tensor(result)

Posso usar a biblioteca em aplicações de produção?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Sim, a biblioteca é projetada para ser usada em aplicações de produção. No entanto, como qualquer software, é recomendável testar extensivamente antes de implantar em produção.

A biblioteca é adequada para sistemas embarcados?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Sim, a biblioteca é adequada para sistemas embarcados baseados em RISC-V. Ela foi projetada para ser leve e eficiente, com poucas dependências externas. A implementação de fallback também permite que a biblioteca seja usada em sistemas embarcados sem suporte a RVV.

Existe suporte comercial disponível?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Atualmente, não oferecemos suporte comercial oficial. No entanto, você pode entrar em contato com a equipe de desenvolvimento para discutir possíveis arranjos de suporte personalizado.

Perguntas sobre Desenvolvimento
-----------------------------

Como posso adicionar uma nova operação à biblioteca?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Para adicionar uma nova operação:

1. Implemente a versão RVV e a versão de fallback da operação em C++
2. Adicione testes para a nova operação
3. Adicione bindings Python para a operação
4. Atualize a documentação
5. Envie um pull request

Consulte o arquivo CONTRIBUTING.md para mais detalhes.

Como posso reportar um bug?
^^^^^^^^^^^^^^^^^^^^^^^^^

Você pode reportar bugs abrindo uma issue no GitHub: https://github.com/JPEDROPS092/sop/issues

Inclua as seguintes informações:
- Descrição detalhada do problema
- Passos para reproduzir o bug
- Ambiente (sistema operacional, versão do compilador, hardware)
- Saída de erro (se aplicável)
- Código de exemplo mínimo que reproduz o problema

Existe um roadmap para futuras versões?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Sim, estamos planejando várias melhorias para futuras versões:

1. Suporte a mais tipos de dados (int8, int16, float16)
2. Mais operações de ML (transformers, RNNs)
3. Integração mais profunda com frameworks de ML
4. Otimizações específicas para diferentes implementações de RVV
5. Suporte a computação paralela multi-core

Consulte o arquivo ROADMAP.md no repositório para mais detalhes.